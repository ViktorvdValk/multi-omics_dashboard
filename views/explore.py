"""Subset explorer: select samples by criteria, then (1) plot a dataset on them coloured by any variable and
(2) plot two variables against each other."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

import analysis as A
import explore as X
import skin_engine as E
import skin_plots as P
from views import _common as C

AXES = {"PCo1 vs PCo2": (0, 1), "PCo1 vs PCo3": (0, 2), "PCo2 vs PCo3": (1, 2)}


def pick_var(prefix, datasets, base_name, base_md, default="severity", from_base=False) -> X.Var:
    """Row of select boxes that defines one variable: metadata, or a feature / summary index of any dataset.
    default: the metadata variable shown first; from_base: start on the base dataset's features instead."""
    labels = {ds.label: n for n, ds in datasets.items()}
    cols = st.columns([1.2, 1.2, 1.1, 2.6, 1.5])
    sources = [X.META] + list(labels)
    source = cols[0].selectbox("From", sources, key=f"{prefix}_src_{base_name}",
                               index=sources.index(datasets[base_name].label) if from_base else 0)
    if source == X.META:
        names = X.meta_variables(base_md, datasets)
        name = cols[3].selectbox("Variable", names, index=names.index(default) if default in names else 0,
                                 format_func=lambda v: X.META_LABELS.get(v, v), key=f"{prefix}_meta")
        return X.Var(X.META, name)
    ds = datasets[labels[source]]
    sites = ([X.SAME] if ds.name == base_name else []) + list(ds.sites)
    site = cols[1].selectbox("Site", sites, key=f"{prefix}_site_{ds.name}",
                             format_func=lambda x: x if x == X.SAME else ds.site_title(x),
                             help="'same sample': the value of the plotted sample itself. A site: the value of the "
                                  "same patient at the same timepoint at that site (lesional: the most severe lesion).")
    kinds = ["feature"] + (["index"] if ds.derived is not None and ds.derived.shape[1] else [])
    kind = cols[2].selectbox("Kind", kinds, key=f"{prefix}_kind_{ds.name}",
                             format_func={"feature": ds.feature_label, "index": "summary index"}.get)
    if kind == "index":
        name = cols[3].selectbox("Index", list(ds.derived.columns), key=f"{prefix}_idx_{ds.name}")
        return X.Var(ds.name, name, "index", site)
    feats = list(ds.mat.mean().sort_values(ascending=False).index)
    name = cols[3].selectbox(f"{ds.feature_label.capitalize()} (highest mean first)", feats, key=f"{prefix}_feat_{ds.name}")
    tr = X.TRANSFORMS[ds.kind]
    transform = cols[4].selectbox("As", list(tr), format_func=tr.get, key=f"{prefix}_tr_{ds.name}")
    return X.Var(ds.name, name, "feature", site, transform)


def _criteria(datasets, base_name, base_md):
    """Criteria editor. Returns (mask over base_md, list of descriptions)."""
    st.session_state.setdefault("ex_ncrit", 0)
    b1, b2, b3 = st.columns([1, 1, 4])
    if b1.button("Add criterion", key="ex_add"):
        st.session_state["ex_ncrit"] += 1
    if b2.button("Remove last", key="ex_del", disabled=st.session_state["ex_ncrit"] == 0):
        st.session_state["ex_ncrit"] -= 1
    n = st.session_state["ex_ncrit"]
    how = "all"
    if n > 1:
        how = b3.radio("A sample must meet", ["all", "any"], horizontal=True, key="ex_how",
                       format_func={"all": "all criteria", "any": "at least one criterion"}.get)
    masks, texts = [], []
    for i in range(n):
        with st.container(border=True):
            var = pick_var(f"ex_c{i}", datasets, base_name, base_md, default="local_severity")
            s = X.resolve(var, base_name, base_md, datasets)
            label = var.label(datasets)
            uid = f"ex_c{i}_{var.source}_{var.name}_{var.site}_{var.transform}"
            if X.is_numeric(s):
                v = pd.to_numeric(s, errors="coerce").dropna()
                if v.empty:
                    st.caption("No sample in the selection has a value for this variable.")
                    continue
                c1, c2, c3, c4 = st.columns([1, 1.3, 1.3, 3])
                op = c1.selectbox("Condition", X.NUMERIC_OPS, key=f"{uid}_op")
                whole = bool((v == v.round()).all())
                kw = dict(step=1.0 if whole else None, format="%g")
                if op == "between":
                    value = (c2.number_input("from", value=float(v.min()), key=f"{uid}_lo", **kw),
                             c3.number_input("to", value=float(v.max()), key=f"{uid}_hi", **kw))
                else:
                    start = v.mode().iloc[0] if op in ("=", "≠") else v.min() if op in ("≥", ">") else v.max()
                    value = c2.number_input("Value", value=float(start), key=f"{uid}_{op}_val", **kw)
                c4.caption(f"Values among the samples above: {v.min():g} to {v.max():g} "
                           f"({v.nunique()} distinct, {int(s.isna().sum())} sample(s) without a value)")
            else:
                levels = X.text_levels(s, var.name)
                c1, c2 = st.columns([1, 4])
                op = c1.selectbox("Condition", X.TEXT_OPS, key=f"{uid}_op")
                value = c2.multiselect("Values", levels, key=f"{uid}_vals")
                if not value:
                    st.caption("Choose at least one value; until then this criterion is ignored.")
                    continue
            m = X.criterion_mask(s, op, value)
            masks.append(m)
            texts.append(X.describe(label, op, value) + f"  ({int(m.sum())} samples)")
    return X.combine(masks, how, base_md.index), texts, how


def _ordination(ds, md, datasets, base_name):
    if len(md) < 3:
        st.info("At least three samples are needed for an ordination.")
        return
    c1, c2 = st.columns(2)
    if ds.kind == "compositional":
        metric = c1.selectbox("Distance", ["braycurtis", "aitchison"], key="ex_metric")
        mat = ds.mat.loc[md.index]
        keep = mat.columns[(mat > 0).any(axis=0)]
        D = E.distance_matrices(mat[keep], ds.values.loc[md.index, keep])[metric]
        title = f"{ds.label}: {metric} PCoA of the selected samples"
    else:
        c1.caption("Distance: Euclidean on z-scored ln values (features two samples both have).")
        D = A.euclidean(ds.values.loc[md.index])
        title = f"{ds.label}: PCA of the selected samples"
    axes = AXES[c2.selectbox("Axes", list(AXES), key="ex_axes")]
    st.markdown("**Colour by**")
    var = pick_var("ex_col", datasets, base_name, md)
    colour = X.resolve(var, base_name, md, datasets)
    coords, expl = E.pcoa(D)
    if coords.shape[1] <= max(axes):
        st.info("The selected samples do not differ enough for these axes.")
        return
    a, b = axes
    label = var.name if var.source == X.META and var.name == "severity" else var.label(datasets)
    C.plot(P.scatter_figure(coords[:, a], coords[:, b], md, colour, label, title=title,
                            xlab=f"PCo{a + 1} ({expl[a]:.1%})", ylab=f"PCo{b + 1} ({expl[b]:.1%})"), key="ex_pcoa")
    st.caption("The ordination is calculated on the selected samples only. Numbers run from blue (lowest) to red "
               "(highest); healthy volunteers are green when the colour is a severity score; grey = no value.")


def _two_variables(md, datasets, base_name):
    st.markdown("**Horizontal axis**")
    vx = pick_var("ex_x", datasets, base_name, md, default="global_severity")
    st.markdown("**Vertical axis**")
    vy = pick_var("ex_y", datasets, base_name, md, from_base=True)
    vc = None
    if st.checkbox("Colour by a third variable", value=True, key="ex_usecol"):
        vc = pick_var("ex_c3", datasets, base_name, md)
    x, y = X.resolve(vx, base_name, md, datasets), X.resolve(vy, base_name, md, datasets)
    lx, ly = vx.label(datasets), vy.label(datasets)
    colour = X.resolve(vc, base_name, md, datasets) if vc else None
    lc = ("severity" if vc and vc.source == X.META and vc.name == "severity" else vc.label(datasets)) if vc else ""
    ok = x.notna() & y.notna()
    if ok.sum() < 2:
        st.info("Fewer than two selected samples have both variables.")
        return
    d, xs, ys = md[ok], x[ok], y[ok]
    cs = colour[ok] if colour is not None else None
    repeated = d["patient"].duplicated().any()
    note = (f"{len(d)} samples of {d['patient'].nunique()} patients have both variables"
            + (f"; {int((~ok).sum())} selected sample(s) lack one of them" if (~ok).any() else "") + ".")
    numx, numy = X.is_numeric(xs), X.is_numeric(ys)
    if numx and numy:
        c1, c2 = st.columns(2)
        logx = c1.checkbox("Logarithmic horizontal axis", key="ex_logx", disabled=bool((xs <= 0).all()))
        logy = c2.checkbox("Logarithmic vertical axis", key="ex_logy", disabled=bool((ys <= 0).all()))
        fig = P.scatter_figure(xs.values, ys.values, d, cs, lc, xlab=lx, ylab=ly)
        if logx:
            fig.update_xaxes(type="log")
        if logy:
            fig.update_yaxes(type="log")
        C.plot(fig, key="ex_scatter")
        if xs.nunique() > 1 and ys.nunique() > 1 and len(d) >= 4:
            rho, p = stats.spearmanr(xs.values, ys.values)
            note += f" Spearman rho = {rho:.2f}, p = {p:.3g}."
        if logx or logy:
            note += " Values of zero or below are not shown on a logarithmic axis."
    elif numx != numy:
        cat, num, lcat, lnum, vcat = (ys, xs, ly, lx, vy) if numx else (xs, ys, lx, ly, vx)
        levels = X.text_levels(cat, vcat.name)
        C.plot(P.category_figure(cat.astype(str), num.values, d, levels, cs, lc, xlab=lcat, ylab=lnum), key="ex_cat")
        groups = [num[(cat.astype(str) == l).values].values for l in levels]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) == 2 and len(set(np.concatenate(groups))) > 1:
            note += f" Mann-Whitney U test: p = {stats.mannwhitneyu(*groups, alternative='two-sided').pvalue:.3g}."
        elif len(groups) > 2 and len(set(np.concatenate(groups))) > 1:
            note += f" Kruskal-Wallis test: p = {stats.kruskal(*groups).pvalue:.3g}."
        if numy is False:
            note += " The text variable is drawn on the horizontal axis."
    else:
        tab = pd.crosstab(ys.astype(str), xs.astype(str)).reindex(index=X.text_levels(ys, vy.name),
                                                                  columns=X.text_levels(xs, vx.name), fill_value=0)
        tab.index.name, tab.columns.name = ly, lx
        st.caption("Both variables are text: number of samples per combination.")
        st.dataframe(tab, **C.STRETCH_DF)
    st.caption(note + (" Several samples come from the same patient, so the p-value is only a rough guide."
                       if repeated and "p =" in note else ""))
    return {lx: x, ly: y, **({lc: colour} if colour is not None else {})}


def render():
    st.title("Subset explorer")
    datasets = st.session_state.get("bundle", {}).get("datasets", {})
    if not datasets:
        st.info("Load data in **Data setup** first.")
        return
    st.caption("Choose which samples to look at, narrow them down with criteria (for example local severity = 8), "
               "then plot the dataset on those samples or plot two variables against each other. A variable of "
               "another dataset is matched on patient and timepoint.")

    st.subheader("1. Samples")
    labels = {ds.label: n for n, ds in datasets.items()}
    c1, c2, c3, c4 = st.columns([1.3, 2, 2, 1.4])
    base_name = labels[c1.selectbox("Dataset", list(labels), key="ex_base")]
    ds = datasets[base_name]
    sites = c2.multiselect("Sites", ds.sites, default=ds.sites, key=f"ex_sites_{base_name}", format_func=ds.site_title)
    tp_opts = E.timepoint_levels(ds.md)
    tps = c3.multiselect("Timepoints", tp_opts, default=tp_opts, key=f"ex_tp_{base_name}",
                         help="Healthy volunteers who were sampled once are kept whatever is selected here.")
    md = ds.md[ds.md["site"].isin(sites) & (ds.md["timepoint"].isna() | ds.md["timepoint"].isin(tps)
                                            | E.timeless_healthy(ds.md))]      # healthy sampled once: always kept
    if E.multiple_lesions(md):
        if c4.radio("Lesions", ["most severe", "both"], key="ex_lesions", horizontal=True) == "most severe":
            md = E.most_severe_lesion(md)
    if md.empty:
        st.warning("No samples for this choice.")
        return

    st.subheader("2. Criteria")
    mask, texts, how = _criteria(datasets, base_name, md)
    sel = md[mask.values]
    joiner = " AND " if how == "all" else " OR "
    st.markdown(f"**{len(sel)} of {len(md)} samples selected** ({sel['patient'].nunique()} patients)"
                + (": " + joiner.join(t.split("  (")[0] for t in texts) if texts else " (no criteria: all samples)"))
    if sel.empty:
        st.warning("No sample meets the criteria.")
        return

    t_ord, t_two, t_tab = st.tabs([f"Plot {ds.label.lower()}", "Plot two variables", "Selected samples"])
    with t_ord:
        _ordination(ds, sel, datasets, base_name)
    with t_two:
        used = _two_variables(sel, datasets, base_name) or {}
    with t_tab:
        cols = [c for c in ("patient", "study", "group", "severity", "site", "timepoint", "lesion", "local_severity",
                            "global_severity", "sample") if c in sel]
        out = sel[cols].assign(timepoint=sel["timepoint"].astype(object))
        for name, s in used.items():
            out[name] = s.reindex(out.index).values
        C.table(out)
        st.download_button("Download the selected samples (CSV)", out.to_csv(index=False), "selected_samples.csv",
                           key="ex_dl")
