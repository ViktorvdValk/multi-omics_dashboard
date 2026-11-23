"""Integrated analysis: compare ordination spaces, colour one by variables of another, combine spaces."""
from __future__ import annotations

import itertools

import pandas as pd
import streamlit as st

import datasets as DS
import integration as I
import skin_engine as E
from views import _common as C

AXES = {"PCo1 vs PCo2": (0, 1), "PCo1 vs PCo3": (0, 2), "PCo2 vs PCo3": (1, 2)}


def _sidebar(datasets):
    md = pd.concat([ds.md for ds in datasets.values()])
    with st.sidebar:
        st.header("Filters")
        tp_opts = E.timepoint_levels(md)
        tps = st.multiselect("Timepoints", tp_opts, default=tp_opts, key="int_tp",
                             help="One point is one patient at one timepoint. Select a single timepoint for one "
                                  "point per patient.")
        opts = sorted(md["study"].unique())
        studies = st.multiselect("Studies", opts, default=opts, key="int_studies")
        opts = sorted(md["group"].unique())
        groups = st.multiselect("Disease groups", opts, default=opts, key="int_groups")
        opts = E.severity_levels(md)
        sev = st.multiselect("Severity / stage", opts, default=opts, key="int_sev")
        lesions = "L1"
        if any(E.multiple_lesions(ds.md) for ds in datasets.values()):
            lesions = st.radio("Patients with two lesions", list(I.LESION_MODES), format_func=I.LESION_MODES.get,
                               key="int_lesions",
                               help="L1 = the most severe lesion of the visit, L2 = the less severe one. With both, "
                                    "L2 is a point of its own, matched to the L2 samples of other modalities only; "
                                    "the permutation tests keep a patient's L1 and L2 together. Nothing is averaged.")
        st.header("Spaces and tests")
        metric = st.selectbox("Microbiome distance", ["aitchison", "braycurtis"], key="int_metric")
        k = st.slider("PCoA axes used by Procrustes", 2, 10, 5, key="int_k")
        n_perm = st.number_input("Permutations", 99, 9999, 999, step=100, key="int_nperm")
        within = st.checkbox("Adjust for study", value=len(studies) > 1, key="int_within",
                             help="Permutations stay within a study (and study is a covariate for the variable "
                                  "test), so a difference between studies is not read as an association.")
    f = {"studies": studies, "groups": groups, "severities": sev, "timepoints": tps, "lesions": lesions}
    return f, metric, int(k), int(n_perm), bool(within)


def _colour_picker(views, spaces, meta):
    """Colour / test variable from the metadata, from any view (feature or summary index) or an ordination axis."""
    st.markdown("**Colour by**")
    c1, c2, c3 = st.columns([2, 1.3, 3])
    source = c1.selectbox("Source", ["Metadata"] + list(spaces), key="int_csource",
                          help="A variable measured in one modality can colour the ordination of another.")
    if source == "Metadata":
        cols = ["severity", "group", "study", "timepoint", "pasi", "local_severity", "global_severity"]
        return "Metadata", "meta", c3.selectbox("Variable", [c for c in cols if c in meta and meta[c].notna().any()],
                                                key="int_cmeta",
                                                help="local_severity: the local severity of the lesion of that point (L1 or L2). "
                                                     "Severity scores run blue to red; healthy volunteers are green.")
    kinds = (list(I.variable_options(views[source])) if source in views else []) + ["ordination axis"]
    kind = c2.selectbox("Kind", kinds, key=f"int_ckind_{source}")
    if kind == "ordination axis":
        return source, "axis", c3.selectbox("Axis", ["PC1", "PC2", "PC3"], key="int_caxis")
    names = I.variable_options(views[source])[kind]
    return source, kind, c3.selectbox(f"{views[source].feature_label.capitalize()} (highest mean first)"
                                      if kind == "feature" else "Index", names, key=f"int_cname_{source}_{kind}")


def _default(names, wanted, n):
    picked = [next((x for x in names if w in x), None) for w in wanted]
    picked = [p for p in dict.fromkeys(picked) if p]
    return (picked + [x for x in names if x not in picked])[:n] if len(picked) < n else picked[:n]


def render():
    st.title("Integrated analysis")
    datasets = st.session_state.get("bundle", {}).get("datasets", {})
    if len(datasets) < 2:
        st.info("Load at least two modalities in **Data setup** to compare or combine them.")
        return
    f, metric, k, n_perm, within = _sidebar(datasets)
    views = I.build_views(datasets, f)
    views = {n: v for n, v in views.items() if len(v.values) >= 4}
    if len(views) < 2:
        st.warning("Fewer than two views have at least four units with the current filters.")
        return
    meta = I.units_meta(views)
    spaces = {n: I.view_distance(v, metric) for n, v in views.items()}
    saved = st.session_state.setdefault("combined_spaces", {})
    skipped = []
    for name, spec in saved.items():
        parts = {p: spaces[p] for p in spec["parts"] if p in views}
        if len(parts) == len(spec["parts"]):
            D, info = I.fuse(parts, spec["min"])
            if len(D) >= 4:
                spaces[name] = D
                continue
        skipped.append(name)
    if skipped:
        st.caption(f"Saved combined space(s) not available with the current filters: {', '.join(skipped)}")
    n_l2 = int(meta["lesion"].astype(object).isin([f"L{i}" for i in range(2, 10)]).sum()) if "lesion" in meta else 0
    st.caption(f"One point = one sample: one patient at one timepoint"
               + (f", and a patient's second lesion (L2) as a point of its own ({n_l2} L2 points)"
                  if f["lesions"] == "both" else "")
               + f". Modalities are matched on study, patient and timepoint"
               + (" (and lesion)" if f["lesions"] == "both" else "")
               + f"; nothing is averaged. {len(views)} views from {len(datasets)} modalities, {len(meta)} units.")
    for v in views.values():
        if v.note:
            st.warning(v.note)

    spec = _colour_picker(views, spaces, meta)
    colour, label = I.get_variable(spec, views, spaces, meta)
    imputations = tuple(ds.imputation for ds in datasets.values())
    state = (str(sorted(f.items())), metric, k, n_perm, within, spec, imputations)

    t_cov, t_cmp, t_comb = st.tabs(["Coverage", "Compare spaces", "Combined space"])

    with t_cov:
        st.subheader("Patients per study, timepoint and modality")
        C.table(DS.coverage(datasets))
        st.subheader("Units shared between views")
        st.caption("Diagonal: units in the view. Two spaces can only be compared on the units they share.")
        st.dataframe(I.unit_coverage(views), **C.STRETCH_DF)

    with t_cmp:
        names = list(spaces)
        chosen = st.multiselect("Spaces to show side by side (2 to 4)", names, max_selections=4, key="int_spaces",
                                default=_default(names, ["Skin microbiome · lesional", "Gut", "Ceramides · lesional"], 3))
        c1, c2 = st.columns(2)
        shared = c1.checkbox("Only units present in all selected spaces", key="int_shared")
        axes = AXES[c2.selectbox("Axes", list(AXES), key="int_axes")]
        if len(chosen) < 1:
            st.info("Select at least one space.")
        else:
            sel = {n: spaces[n] for n in chosen}
            C.plot(I.spaces_figure(sel, meta, colour, label, axes=axes, shared_only=shared), key="int_side")
            st.caption("Hollow grey markers: the unit has no value for the colour variable (modality not measured).")
            if st.button("Test", type="primary", key="int_test",
                         help="Does the colour variable explain each space, and do the spaces agree with each other?"):
                rows = [{"space": n, "variable": label, **I.association(D, colour, meta, within, n_perm)}
                        for n, D in sel.items()]
                pairs = []
                for a, b in itertools.combinations(chosen, 2):
                    pr = I.protest(sel[a], sel[b], meta, k, n_perm, within)
                    mn = I.mantel(sel[a], sel[b], meta, "spearman", n_perm, within)
                    pairs.append({"space A": a, "space B": b, "shared units": pr["n"],
                                  "Procrustes r": pr.get("procrustes r"), "Procrustes p": pr.get("p"),
                                  "Mantel r": mn.get("mantel r"), "Mantel p": mn.get("p"), "note": pr.get("note", "")})
                assoc, pairs = pd.DataFrame(rows), pd.DataFrame(pairs)
                for tab, cols in ((assoc, {"p": "q (BH)"}), (pairs, {"Procrustes p": "Procrustes q (BH)",
                                                                     "Mantel p": "Mantel q (BH)"})):
                    for c, q in cols.items():
                        if len(tab) and c in tab:
                            tab[q] = E.adjust_p(tab[c].astype(float).values)
                st.session_state["int_results"] = {"key": (state, tuple(chosen)), "assoc": assoc, "pairs": pairs}
            res = st.session_state.get("int_results")
            if res and res["key"] == (state, tuple(chosen)):
                st.markdown(f"**Does *{label}* explain each space?** (distance-based regression, R² = share of variance)")
                C.table(res["assoc"])
                st.caption("'within patient': the variable changes between a patient's timepoints, so it is tested "
                           "on those changes (patients with one timepoint add nothing). 'whole patients': the "
                           "variable is fixed per patient and all timepoints of a patient are kept together.")
                if len(res["pairs"]):
                    st.markdown("**Do the spaces agree?** (on shared units)")
                    C.table(res["pairs"])
                    st.caption("Procrustes r and Mantel r near 0: patients that are similar in one space are not "
                               "similar in the other. q: Benjamini-Hochberg over the rows of the table.")
            if len(chosen) >= 2:
                with st.expander("Procrustes overlay of two spaces"):
                    c1, c2 = st.columns(2)
                    a = c1.selectbox("Reference", chosen, key="int_pa")
                    b = c2.selectbox("Rotated onto it", [x for x in chosen if x != a], key="int_pb")
                    if len(spaces[a].index.intersection(spaces[b].index)) >= 6:
                        C.plot(I.procrustes_figure(spaces[a], spaces[b], a, b, meta, colour, label, k), key="int_proc")
                    else:
                        st.info("Fewer than six shared units.")

    with t_comb:
        base = list(views)
        parts = st.multiselect("Views to combine", base, key="int_parts",
                               default=_default(base, ["Skin microbiome · lesional", "Ceramides · lesional"], 2))
        if len(parts) < 2:
            st.info("Select at least two views.")
            return
        need = len(parts)
        if len(parts) > 2:
            need = st.slider("A unit needs at least this many of the selected views", 2, len(parts), len(parts),
                             key="int_min", help="All of them (default) is exact. Fewer keeps more units but the "
                                                  "distance between two units then uses only the views both have.")
        D, info = I.fuse({p: spaces[p] for p in parts}, need)
        if len(D) < 4:
            st.warning(f"Only {len(D)} units have the required views. Relax the filters or combine fewer views.")
            return
        name = "Combined: " + " + ".join(parts)
        C.plot(I.spaces_figure({name: D}, meta, colour, label, height=560), key="int_comb")
        st.caption(f"{info['n_units']} units. Each view is scaled to the same total variance before combining"
                   + ("." if info["complete"] else "; units may miss some views, so this space is approximate.")
                   + (f" {info['dropped_no_overlap']} unit(s) without any view in common with another were dropped."
                      if info["dropped_no_overlap"] else ""))
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**What the combined space reflects**")
            C.table(I.contributions({p: spaces[p] for p in parts}, D))
        with c2:
            st.markdown("**Use it as a space**")
            st.caption("A saved combined space appears under *Compare spaces* and as a colour source "
                       "(its axes), so it can be compared with any other view.")
            label_in = st.text_input("Name", name, key="int_combname")
            if st.button("Save combined space", key="int_save"):
                saved[label_in] = {"parts": list(parts), "min": need}
                st.rerun()
            for s in list(saved):
                if st.button(f"Remove '{s}'", key=f"int_rm_{s}"):
                    del saved[s]
                    st.rerun()
        if st.button("Test the colour variable in the combined space", key="int_combtest"):
            st.session_state["int_comb_result"] = {"key": (state, tuple(parts), need),
                                                   "res": I.association(D, colour, meta, within, n_perm)}
        res = st.session_state.get("int_comb_result")
        if res and res["key"] == (state, tuple(parts), need):
            C.table(pd.DataFrame([{"space": name, "variable": label, **res["res"]}]))
