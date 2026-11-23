"""Shared page for concentration-type modalities (ceramides, energy metabolites)."""
from __future__ import annotations

from dataclasses import replace

import streamlit as st

import analysis as A
import datasets as DS
import skin_engine as E
import skin_plots as P
from views import _common as C


def missing_values_control(ds, key):
    """Sidebar choice of how zero / missing values are handled; applies to this modality on every page."""
    n_missing = int(ds.log_observed.isna().values.sum())
    with st.sidebar:
        st.header("Zero and missing values")
        method = st.selectbox(f"{n_missing} value(s) ({n_missing / max(ds.log_observed.size, 1):.1%})",
                              list(DS.IMPUTATIONS), index=list(DS.IMPUTATIONS).index(ds.imputation),
                              format_func=DS.IMPUTATIONS.get, key=f"{key}_impute", disabled=n_missing == 0,
                              help="A zero is read as 'below the detection limit', not as a concentration of zero. "
                                   "Leave out: each feature is analysed on the samples that have a value. "
                                   "Half of the smallest value suits values that are missing because they are low; "
                                   "nearest neighbours suits values missing by chance. The choice also applies on "
                                   "the Integrated page.")
    ds.set_imputation(method)


def render(ds, key, extra_tab=None, prepare=None):
    """extra_tab = (title, function(ds, md, mat, S)) adds a modality-specific tab after the overview.
    prepare(ds) runs after the missing-value choice (e.g. to rebuild indices that depend on the values)."""
    st.title(ds.label)
    S, factor = C.sidebar_settings(ds, key)
    missing_values_control(ds, key)
    if prepare:
        prepare(ds)
    with st.sidebar:
        min_prev = st.slider(f"Test a {ds.feature_label} when measured in at least", 0.0, 1.0, 0.5, 0.05,
                             key=f"{key}_minprev", help="Share of the samples of a site with a value above zero.")
    S = replace(S, min_prev=float(min_prev))
    md = E.filter_md(ds.md, S)      # applies the lesion selection (S.lesions) for statistics and plots
    if md.empty:
        st.warning("No samples match the selected filters.")
        return
    mat, values = ds.mat.loc[md.index], ds.values.loc[md.index]
    levels = C.factor_levels(md, factor, S.studies)
    strata = E.strata_for(set(md["site"]))
    sev_colors = E.severity_colors(E.severity_levels(md))
    colors = (sev_colors if factor == "severity" else P.scale_colors(levels) if factor == "local_severity"
              else P.categorical_colors(levels))
    po = C.plot_options(md, key)
    grad = po["grad"]
    by_ls = po["group"] or factor == "local_severity"
    box = dict(keep_lesions=S.lesions == "both", x_map=po["map"] if by_ls else None, grad=grad,
               site_titles=ds.site_titles or None,
               average=S.timepoint_mode == "average")
    box_levels, box_colors = (po["levels"], P.scale_colors(po["levels"])) if by_ls else (levels, colors)

    names = ["Overview"] + ([extra_tab[0]] if extra_tab else []) + ["Profile", "Features", "Ordination", "Statistical analysis"]
    tabs = dict(zip(names, st.tabs(names)))
    with tabs["Overview"]:
        C.overview(md, int((mat > 0).any().sum()), f"{ds.feature_label.capitalize()}s")
        measured = (mat > 0).groupby(md["site"].values).mean().T.round(2)
        if (measured < 1).any().any():
            st.caption(f"Share of samples with a value above zero, per {ds.feature_label} and site "
                       f"(only those below 100% are listed):")
            C.table(measured[(measured < 1).any(axis=1)].rename_axis(ds.feature_label).reset_index())
        for note in ds.notes:
            st.caption(note)
    if extra_tab:
        with tabs[extra_tab[0]]:
            extra_tab[1](ds, md, mat, S)
    derived = ds.derived.loc[md.index] if ds.derived is not None and ds.derived.shape[1] else None
    with tabs["Profile"]:
        pf = "severity" if factor in ("site", "local_severity") else factor
        st.caption(f"Mean z-score of ln values per site and {pf}. Red = above the overall mean.")
        C.plot(A.profile_heatmap(values, md, pf, C.factor_levels(md, pf, S.studies)), key=f"{key}_profile")
    with tabs["Features"]:
        pool = {f"{ds.feature_label.capitalize()}s (ln)": (values, list(mat.mean().sort_values(ascending=False).index))}
        if derived is not None:
            pool["Summary indices"] = (derived, list(derived.columns))
        c1, c2 = st.columns([1, 2])
        kind = c1.selectbox("Show", list(pool), key=f"{key}_featkind")
        frame, options = pool[kind]
        feat = c2.selectbox("Variable", options, key=f"{key}_feat")
        fig = C.unit_boxes(md, frame[[feat]], strata, factor, box_levels, box_colors, sev_colors,
                           note="One point per patient, site and timepoint; samples without a value are not shown.", **box)
        if fig is not None:
            C.plot(fig, key=f"{key}_box")
        if {"lesional", "non-lesional"} <= set(md["site"]) and kind != "Summary indices":
            st.subheader(f"Paired {ds.site_title('lesional')} / {ds.site_title('non-lesional')}")
            C.plot(P.paired_figure(E.most_severe_lesion(md), mat.where(mat > 0), feat, E.severity_levels(md), sev_colors,
                                   pseudo=0, timepoints=list(S.timepoints), ylabel=f"log10({feat})", grad=grad),
                   key=f"{key}_paired")
    with tabs["Ordination"]:
        if len(md) >= 3:
            color_by = st.selectbox("Colour by", ["severity", "group", "study", "timepoint"], key=f"{key}_pcoacol",
                                    disabled=grad is not None,
                                    help="Severity scores are chosen under 'Plots' in the sidebar; while one is "
                                         "selected there it is used here too.")
            coords, expl = E.pcoa(A.euclidean(values))
            if coords.shape[1] >= 2:
                C.plot(P.pcoa_figure(coords, expl, md, sev_colors, color_by=color_by, grad=grad,
                                     title="PCA (Euclidean distance on z-scored ln values)"), key=f"{key}_pcoa")
        else:
            st.info("At least three samples are needed.")
    with tabs["Statistical analysis"]:
        def run(progress):
            return C.relabel_sites(analyse(progress), ds.site_titles)

        def analyse(progress):
            if factor == "site":
                return A.run_site_analysis(md, values, mat, S, "abundance", derived=derived, progress=progress)
            if factor == "local_severity":
                return A.run_local_severity_analysis(ds.md, ds.values, ds.mat, S, "abundance",
                                                     derived=ds.derived, progress=progress)
            return A.run_abundance_analysis(md, values, mat, S, factor=factor, levels_all=levels,
                                            ordered=(factor == "severity"), derived=derived, progress=progress)
        extra = (ds.imputation, ",".join(derived.columns) if derived is not None else "")
        methods = (A.local_severity_methods(S, "abundance", ds.feature_label) if factor == "local_severity"
                   else (A.SITE_METHODS + "\n" if factor == "site" else "")
                   + A.methods_text(S, factor, ds.feature_label, ds.imputation))
        C.results_section(key, S, factor, run, "log2 fold change", extra_key=extra, methods=methods)
