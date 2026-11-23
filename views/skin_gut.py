"""Skin and gut microbiome pages (one function, called with 'skin' or 'gut')."""
from __future__ import annotations

import streamlit as st

import skin_engine as E
import skin_plots as P
import analysis as A
from views import _common as C


def render(compartment):
    ds = C.get_dataset(compartment, f"{compartment.capitalize()} microbiome")
    if ds is None:
        return
    st.title(ds.label)
    rank = st.session_state["bundle"].get("rank", "taxon")
    S, factor = C.sidebar_settings(ds, compartment, skin_types=(compartment == "skin"))
    md = E.filter_md(ds.md, S)
    if md.empty:
        st.warning("No samples match the selected filters.")
        return
    mat, clr = ds.mat.loc[md.index], ds.values.loc[md.index]
    levels = C.factor_levels(md, factor, S.studies)
    strata = E.strata_for(set(md["site"]))
    sev_colors = E.severity_colors(E.severity_levels(md))
    colors = (sev_colors if factor == "severity" else P.scale_colors(levels) if factor == "local_severity"
              else P.categorical_colors(levels))
    has_tp = md["timepoint"].notna().any()
    po = C.plot_options(md, compartment)
    grad = po["grad"]
    by_ls = po["group"] or factor == "local_severity"
    box = dict(keep_lesions=S.lesions == "both", x_map=po["map"] if by_ls else None, grad=grad,
               average=S.timepoint_mode == "average")
    box_levels, box_colors = (po["levels"], P.scale_colors(po["levels"])) if by_ls else (levels, colors)

    tabs = st.tabs(["Overview", "Composition", "Diversity", "Taxa", "Statistical analysis"])
    with tabs[0]:
        C.overview(md, int((mat > 0).any().sum()), f"{rank.capitalize()}-level taxa")
        for note in ds.notes:
            st.caption(note)
    with tabs[1]:
        split = st.selectbox("Split by", ["none", "study", "group"] + (["skin_type"] if compartment == "skin" else [])
                             + (["timepoint"] if has_tp else []), key=f"{compartment}_split")
        top_n = st.slider("Top taxa", 5, 30, 12, key=f"{compartment}_topn")
        C.plot(P.composition_figure(mat, md, E.severity_levels(md), top_n=top_n,
                                    split=None if split == "none" else split), key=f"{compartment}_comp")
    with tabs[2]:
        st.subheader("Alpha diversity")
        fig = C.unit_boxes(md, ds.derived.loc[md.index, ["Shannon", "Observed taxa"]], strata, factor, box_levels,
                           box_colors, sev_colors, titles={"Shannon": "Shannon index"}, **box)
        if fig is None:
            st.info("Not enough samples.")
        else:
            C.plot(fig, key=f"{compartment}_alpha")
        st.subheader("PCoA")
        if len(md) >= 3:
            c1, c2 = st.columns(2)
            color_by = c1.selectbox("Colour by", ["severity", "group", "study"] + (["skin_type"] if compartment == "skin" else [])
                                    + (["timepoint"] if has_tp else []), key=f"{compartment}_pcoacol",
                                    disabled=grad is not None,
                                    help="Severity scores are chosen under 'Plots' in the sidebar; while one is "
                                         "selected there it is used here too.")
            metric = c2.selectbox("Distance", ["braycurtis", "aitchison"], key=f"{compartment}_metric")
            coords, expl = E.pcoa(E.distance_matrices(mat, clr)[metric])
            if coords.shape[1] >= 2:
                C.plot(P.pcoa_figure(coords, expl, md, sev_colors, color_by=color_by, title=f"{metric} PCoA", grad=grad),
                       key=f"{compartment}_pcoa")
    with tabs[3]:
        order = mat.mean().sort_values(ascending=False)
        taxa = [t for t in order.index if order[t] > 0]
        taxon = st.selectbox("Taxon (most abundant first)", taxa, key=f"{compartment}_taxon")
        fig = C.unit_boxes(md, clr[[taxon]], strata, factor, box_levels, box_colors, sev_colors,
                           titles={taxon: f"{taxon} (CLR)"}, note="Centred log-ratio values; one point per patient, site and timepoint.",
                           **box)
        if fig is not None:
            C.plot(fig, key=f"{compartment}_taxbox")
        if {"lesional", "non-lesional"} <= set(md["site"]):
            st.subheader("Paired lesional / non-lesional")
            C.plot(P.paired_figure(E.most_severe_lesion(md), mat, taxon, E.severity_levels(md), sev_colors,
                                   timepoints=list(S.timepoints) if has_tp else None, grad=grad),
                   key=f"{compartment}_paired")
    with tabs[4]:
        def run(progress):
            if factor == "site":
                return A.run_site_analysis(md, clr, mat, S, "compositional", derived=ds.derived.loc[md.index],
                                           progress=progress)
            if factor == "local_severity":       # all samples: the within-patient test needs both lesions
                return A.run_local_severity_analysis(ds.md, ds.values, ds.mat, S, "compositional",
                                                     derived=ds.derived, pseudo=ds.pseudo, progress=progress)
            return E.run_analysis(md, mat, clr, S, factor=factor, levels_all=levels, ordered=(factor == "severity"),
                                  progress=progress, pseudo=ds.pseudo)
        methods = (A.local_severity_methods(S, "compositional") if factor == "local_severity" else
                   A.SITE_METHODS + "\n" + E.methods_text(S, factor="site") if factor == "site" else
                   E.methods_text(S, factor=factor))
        C.results_section(compartment, S, factor, run, "log2 FC (CLR)", extra_key=rank, methods=methods)


def skin_page():
    render("skin")


def gut_page():
    render("gut")
