"""Helpers shared by the pages: the loaded data bundle, sidebar filters and result tables."""
from __future__ import annotations

import inspect
import re

import numpy as np
import pandas as pd
import streamlit as st

import skin_engine as E
import skin_plots as P

# Streamlit replaced use_container_width=True by width="stretch"; support both.
STRETCH_PLOT = ({"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters
                else {"use_container_width": True})
STRETCH_DF = ({"width": "stretch"} if "width" in inspect.signature(st.dataframe).parameters
              else {"use_container_width": True})


def plot(fig, key=None):
    st.plotly_chart(fig, key=key, **STRETCH_PLOT)


def table(df, **kw):
    st.dataframe(df, hide_index=True, **STRETCH_DF, **kw)


def get_dataset(name, title):
    """Return the Dataset or None (after telling the user what to load)."""
    ds = st.session_state.get("bundle", {}).get("datasets", {}).get(name)
    if ds is None:
        st.title(title)
        st.info("This modality is not loaded. Open **Data setup** and load it; the other pages keep working "
                "without it.")
    return ds


def factor_levels(md, factor, studies=None):
    if factor == "site":
        return P.site_order(md["site"])
    if factor == "severity":
        return E.severity_levels(md)
    if factor == "timepoint":
        return E.timepoint_levels(md)                     # baseline / pre first -> contrast 'post vs pre'
    if factor == "local_severity":
        return E.local_severity_grouping(md["local_severity"])[1]
    if factor == "study" and studies:
        return [s for s in studies if s in set(md["study"])]
    return sorted(md[factor].dropna().astype(str).unique())


def has_local_severity(md):
    """True if the selected lesional samples carry a usable local severity score."""
    return ("local_severity" in md
            and pd.to_numeric(md.loc[md["site"] == "lesional", "local_severity"], errors="coerce").notna().any())


COLOUR_MODES = {"severity": "severity / stage", "local": "local severity (gradient)",
                "global": "global severity (gradient)", "both": "local (fill) + global (outline)"}


def plot_options(md, key):
    """Sidebar 'Plots' section: how the points of every plot are coloured, and grouping by local severity.
    Returns {available, group, map, levels, grad, mode}; grad is a skin_plots.gradient_spec or None."""
    has = lambda c: c in md and pd.to_numeric(md.loc[md["group"] != E.HEALTHY, c], errors="coerce").notna().any()
    local, glob = has("local_severity"), has("global_severity")
    out = {"available": local, "group": False, "map": None, "levels": [], "grad": None, "mode": "severity"}
    modes = ["severity"] + (["local"] if local else []) + (["global"] if glob else []) + (["both"] if local and glob else [])
    if local:
        out["map"], out["levels"] = E.local_severity_grouping(md["local_severity"])
    if len(modes) == 1:
        return out
    with st.sidebar:
        st.header("Plots")
        mode = st.selectbox("Colour points by", modes, format_func=COLOUR_MODES.get, key=f"{key}_colourmode",
                            help="Applies to the box plots, ordinations and paired plots of this page. Scores run "
                                 "from blue (lowest) to red (highest) with a colour bar; healthy volunteers stay "
                                 "green and samples without a value are grey. Local severity belongs to a lesion, "
                                 "so only lesional samples have it. Global severity belongs to a patient at one "
                                 "timepoint and is never carried over to another timepoint. With both, the fill shows local and the "
                                 "outline global severity.")
        if local:
            out["group"] = st.checkbox("Group by local severity", key=f"{key}_lsgroup",
                                       help="x-axis of the box plots = local severity of the lesion "
                                            + ("(one group per value)." if len(out["levels"]) <= 8 else
                                               f"(range split into {len(out['levels'])} groups).")
                                            + " Only lesional samples have a local severity.")
        out["mode"] = mode
        fill, line = {"severity": (None, None), "local": ("local_severity", None),
                      "global": ("global_severity", None), "both": ("local_severity", "global_severity")}[mode]
        out["grad"] = P.gradient_spec(md, fill, line)
        for col in [c for c in (fill, line) if c] + (["local_severity"] if out["group"] and mode == "severity" else []):
            studies = sorted(md.loc[pd.to_numeric(md[col], errors="coerce").notna() & (md["group"] != E.HEALTHY),
                                    "study"].unique())
            if len(studies) > 1:
                st.caption(f"{P.SCORE_LABELS[col].capitalize()} of studies {', '.join(studies)} is shown on one "
                           f"scale. If the studies score it differently, select one study.")
    return out


def relabel_sites(result, titles):
    """(tables, blocks) of an analysis with site names in the stratum / level / contrast columns replaced by the
    dataset's display names (e.g. lesional -> blister perilesional)."""
    tables, blocks = result
    if not titles:
        return result
    names = sorted(titles, key=len, reverse=True)
    pat = re.compile(r"(?<![\w-])(" + "|".join(map(re.escape, names)) + r")(?![\w-])")
    fix = lambda v: pat.sub(lambda m: titles[m.group(1)], v) if isinstance(v, str) else v
    for k, tab in tables.items():
        if isinstance(tab, pd.DataFrame) and len(tab):
            for c in ("stratum", "level", "contrast", "lo", "hi"):
                if c in tab:
                    tab[c] = tab[c].map(fix)
    return tables, blocks


def plot_grad(md, key):
    """The gradient spec of the page's 'Colour points by' choice (read from the widget state)."""
    mode = st.session_state.get(f"{key}_colourmode", "severity")
    fill, line = {"local": ("local_severity", None), "global": ("global_severity", None),
                  "both": ("local_severity", "global_severity")}.get(mode, (None, None))
    has = lambda c: c in md and pd.to_numeric(md.loc[md["group"] != E.HEALTHY, c], errors="coerce").notna().any()
    return P.gradient_spec(md, fill if fill and has(fill) else None, line if line and has(line) else None)


def unit_boxes(md, values, strata, factor, levels, colors, sev_colors, titles=None, note=None, keep_lesions=False,
               x_map=None, grad=None, average=False, site_titles=None):
    """Box plots with one point per patient, site and timepoint (average=True: per patient and site, the mean of
    the selected timepoints). Between patients: one panel per site, one row per timepoint. Between timepoints:
    x = timepoint, one panel per site. Between sites: x = site, one panel per timepoint.
    x_map: group the x-axis by local severity (function from E.local_severity_grouping) instead of `factor`.
    grad: colour the points by severity score(s) (skin_plots.gradient_spec). keep_lesions: a patient's two
    lesions stay separate points. Returns None without data."""
    by_site = factor == "site" and x_map is None
    per_tp = not average or factor == "timepoint" or by_site       # units keep their timepoint
    tps = E.timepoint_levels(md)
    by_tp = per_tp and not by_site and (x_map is not None or factor != "timepoint") and len(tps) > 1
    ref = E.timeless_healthy(md)                    # healthy sampled once: shown at every timepoint as reference
    ref_patients = set(md.loc[ref.values, "patient"])
    tps = E.timepoint_levels(md[~ref.values]) or tps
    spread = (by_tp or by_site) and bool(ref_patients) and len(tps) > 1
    recs = []
    for stratum in (P.site_order(md["site"]) if by_site else strata):     # x = site: every site on its own
        Y, meta = E.unit_frame(md, values, stratum, factor="timepoint" if per_tp else None,
                               keep_lesions=keep_lesions, shared=not by_site)
        if Y is None:
            continue
        if spread and "timepoint" in meta:
            is_ref = meta["patient"].isin(ref_patients).values
            parts_y, parts_m = [Y[~is_ref]], [meta[~is_ref]]
            for t in tps:
                parts_y.append(Y[is_ref])
                parts_m.append(meta[is_ref].assign(timepoint=t))
            Y, meta = pd.concat(parts_y, ignore_index=True), pd.concat(parts_m, ignore_index=True)
        tp = meta["timepoint"].astype(str) if "timepoint" in meta else None
        for f in values.columns:
            title = (titles or {}).get(f, f)
            d = meta.copy()
            d["y"], d["hover"] = Y[f].values, P._hover(meta, Y[f].values)
            if by_site:
                d["x"], d["facet"], d["row"] = stratum, tp, title
            else:
                d["x"] = ((x_map(meta["local_severity"]) if "local_severity" in meta else np.nan)
                          if x_map is not None else meta[factor].astype(object))
                d["facet"] = stratum
                d["row"] = (title + " · " + tp) if by_tp else title
            recs.append(d)
    if not recs:
        return None
    long = pd.concat(recs, ignore_index=True).dropna(subset=["y", "x"])
    if long.empty:
        return None
    names = [(titles or {}).get(f, f) for f in values.columns]
    rows = [f"{n} · {t}" for n in names for t in tps] if by_tp else names
    rows = [r for r in rows if r in set(long["row"])]
    facets = ([str(t) for t in tps if str(t) in set(long["facet"])] if by_site
              else [s for s in strata if s in set(long["facet"])])
    box_colors = colors if (grad is None or x_map is not None) else {}      # neutral boxes under gradient points
    if note:
        st.caption(note)
    if by_site and site_titles:                       # x = site: use the dataset's site names
        long["x"] = long["x"].map(lambda s: site_titles.get(s, s))
        levels = [site_titles.get(s, s) for s in levels]
        box_colors = {site_titles.get(s, s): c for s, c in box_colors.items()}
    return P.faceted_boxes(long, rows, facets, levels, box_colors, row_titles=rows, grad=grad,
                           point_colors=sev_colors, facet_titles=site_titles,
                           height_per_row=340 if len(rows) <= 2 else 290)


def sidebar_settings(ds, key, skin_types=False):
    """Comparison and sample filters in the sidebar. Returns (Settings, factor)."""
    md = ds.md
    k = lambda name: f"{key}_{name}"
    tp_opts = E.timepoint_levels(md)
    with st.sidebar:
        st.header("Comparison")
        factors = ["severity", "group", "study"] + (["skin_type"] if skin_types else []) + (["timepoint"] if len(tp_opts) > 1 else []) \
            + (["site"] if len(ds.sites) > 1 else []) + (["local_severity"] if has_local_severity(md) else [])
        factor = st.selectbox("Compare by", factors, key=k("factor"),
                              help="Severity, group, study, skin type: between patients, per site. Timepoint: "
                                   "paired within patient, per site. Site: paired within patient, per timepoint. "
                                   "Local severity: lesional samples only, tested as a score between patients and "
                                   "between the two lesions of a patient.")
        tp_mode = "separate"
        if factor != "timepoint" and len(tp_opts) > 1:
            tp_mode = st.selectbox("Timepoints are", ["separate", "average"], key=k("tpmode"),
                                   format_func={"separate": "analysed separately",
                                                "average": "averaged per patient"}.get,
                                   help="Separately: every plot and test is done per timepoint, nothing is mixed. "
                                        "Averaged: one value per patient and site, the mean of the selected "
                                        "timepoints.")
        study_mode = st.selectbox("Studies are", ["separate", "adjust"], key=k("studymode"),
                                  format_func={"separate": "analysed separately", "adjust": "pooled, adjusted for study"}.get)
        group_mode = st.selectbox("Disease groups are", ["separate", "adjust"], key=k("groupmode"),
                                  format_func={"separate": "analysed separately", "adjust": "pooled, adjusted for group"}.get)
        st.header("Sample filters")
        timepoints = st.multiselect("Timepoints", tp_opts, default=tp_opts, key=k("tp"),
                                    help="All timepoints by default. Comparing timepoints is paired: only patients "
                                         "with both timepoints of a pair count. Healthy volunteers who were sampled "
                                         "once are not tied to a timepoint: they are the reference at every "
                                         "timepoint, in the plots and in the statistics.") if tp_opts else []
        opts = sorted(md["study"].unique())
        studies = st.multiselect("Studies", opts, default=opts, key=k("studies"))
        sites = (st.multiselect("Sites", ds.sites, default=ds.sites, key=k("sites"), format_func=ds.site_title)
                 if len(ds.sites) > 1 else ds.sites)
        opts = sorted(md["group"].unique())
        groups = st.multiselect("Disease groups", opts, default=opts, key=k("groups"))
        opts = E.severity_levels(md)
        severities = st.multiselect("Severity / stage", opts, default=opts, key=k("sev"))
        skin_mode, skin_sel = "off", tuple(E.SKIN_TYPES)
        if skin_types and md["skin_type"].notna().any():
            skin_mode = st.selectbox("Skin-type handling", ["off", "restrict", "stratify", "adjust"], key=k("skinmode"),
                                     help="restrict: keep the chosen skin types; stratify: analyse each skin type "
                                          "separately; adjust: skin type as covariate. Nostril samples are kept.")
            present = [t for t in E.SKIN_TYPES if t in set(md["skin_type"].dropna())]
            skin_sel = tuple(st.multiselect("Skin types", present, default=present, key=k("skintypes")))
        if E.multiple_lesions(md):
            lesions = st.radio("Lesions", ["most severe", "both"], horizontal=True,
                               format_func={"most severe": "most severe lesion only (statistics + plots)",
                                            "both": "keep both lesions (plots)"}.get,
                               key=k("lesion"),
                               help="Some studies have two lesions (L1, L2) per patient/timepoint. 'Most severe' "
                                    "keeps the one with the higher local severity score; tests that need one lesion "
                                    "per patient always use the most severe one.")
        else:
            lesions = "most severe"
        st.header("Statistics")
        n_perm = st.number_input("Permutations", 99, 9999, 999, step=100, key=k("nperm"))
        correction = st.selectbox("Multiple-testing correction", ["bh", "holm", "bonferroni"], key=k("corr"))
        q_thr = st.number_input("Significance threshold (q)", 0.001, 0.2, 0.05, step=0.01, format="%.3f", key=k("q"))
    S = E.Settings(studies=tuple(studies), sites=tuple(sites), groups=tuple(groups), severities=tuple(severities),
                   timepoints=tuple(timepoints), skin_mode=skin_mode, skin_types=skin_sel, study_mode=study_mode,
                   group_mode=group_mode, lesions=lesions, correction=correction, q_thr=float(q_thr), n_perm=int(n_perm),
                   timepoint_mode=tp_mode)
    return S, factor


def overview(md, n_features, feature_word):
    left, right = st.columns([1, 2])
    with left:
        st.metric("Samples", len(md))
        st.metric("Patients", md["patient"].nunique())
        st.metric(f"{feature_word} detected", n_features)
    with right:
        cols = ["study", "group", "severity", "site"] + (["timepoint"] if md["timepoint"].notna().any() else [])
        table(md.assign(timepoint=md["timepoint"].astype(object)).groupby(cols, dropna=False).size()
              .rename("samples").reset_index())


def results_section(key, S, factor, run, scale_label, extra_key="", methods=None):
    """Run button + result tables + differential-feature heatmap. `run(progress)` -> (tables, blocks)."""
    st.caption("Permutation tests; this can take a while with many features or permutations.")
    state_key = (S.key(), factor, extra_key)
    if st.button("Run statistical analysis", type="primary", key=f"{key}_run"):
        bar, status = st.progress(0.0), st.empty()

        def progress(fraction, message):
            bar.progress(float(fraction))
            status.write(message)
        tables, blocks = run(progress)
        st.session_state[f"{key}_analysis"] = {"key": state_key, "tables": tables, "blocks": blocks}
    result = st.session_state.get(f"{key}_analysis")
    if not result or result["key"] != state_key:
        st.info("Click **Run statistical analysis** to calculate results for the current settings.")
        return None
    tables, blocks = result["tables"], result["blocks"]
    st.success(f"Analysis complete: {len(blocks)} block(s): {', '.join(b.label for b in blocks) or 'none'}")
    for note in tables.get("notes", []):
        st.warning(note)
    names = {"da": "Features (differential abundance)", "alpha": "Summary indices (alpha diversity, class fractions, ratios)", "beta": "Whole profile (PERMANOVA / PERMDISP)",
             "wdist": "Lesional vs non-lesional distance within patients", "counts": "Units per group"}
    available = [n for n in names if isinstance(tables.get(n), pd.DataFrame) and len(tables[n])]
    if not available:
        st.info("No comparison could be made: fewer than two groups with at least two patients in any block.")
        return tables
    choice = st.selectbox("Results table", available, format_func=names.get, key=f"{key}_table")
    tab = tables[choice]
    if choice != "counts" and st.checkbox(f"Only q < {S.q_thr}", key=f"{key}_onlysig"):
        tab = tab[tab["q"] < S.q_thr]
    table(tab.sort_values("p") if "p" in tab else tab)
    st.download_button("Download this table (CSV)", tab.to_csv(index=False), f"{key}_{choice}_{factor}.csv",
                       key=f"{key}_dl")
    if len(tables.get("da", [])) and blocks:
        block = st.selectbox("Heatmap of differential features for block", [b.label for b in blocks], key=f"{key}_block")
        fig, note = P.da_heatmap(tables["da"], block, S.q_thr, scale_label=scale_label)
        st.caption(note)
        if fig is not None:
            plot(fig, key=f"{key}_heat")
    with st.expander("Methods text"):
        st.text(methods or E.methods_text(S, factor=factor))
    return tables
