"""
integration.py -- compare and combine ordination spaces across modalities.

UNITS  Samples of different modalities are matched on study + patient + timepoint. A unit is therefore one
       patient at one timepoint, and one sample: nothing is averaged. With "both lesions" a patient's second,
       less severe lesion (L2) is a unit of its own (patient + timepoint + L2), matched only to L2 samples of
       other modalities. Sites are never averaged: every site of a modality is its own VIEW with
       one row per unit, e.g. "Skin microbiome · lesional", "Skin microbiome · non-lesional",
       "Skin microbiome · nostril", "Gut microbiome", "Ceramides · lesional", "Energy metabolism · plasma".
       Views share units, so their spaces can be compared and combined, and any variable of one view can
       colour the ordination of another.

SPACE  a distance matrix between units: Aitchison or Bray-Curtis (microbiome), Euclidean on z-scored ln
       values (ceramides, metabolites), or a COMBINED space: D^2 = mean over views of (D_k / s_k)^2 with
       s_k the root-mean-square distance of view k. Each view then carries the same total variance
       (as in multiple factor analysis); with complete data this equals joining the scaled coordinates.

TESTS  PROTEST (Procrustes correlation on the first k PCoA axes) and Mantel for two spaces; distance-based
       regression (PERMANOVA) for "does this variable explain this space?". Permutations move whole
       patients (within study if requested), so the timepoints of one patient are never treated as
       independent. A variable that changes within patients (such as timepoint) is tested within patients.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

import analysis as A
import skin_engine as E
import skin_plots as P

STUDY_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "x", "star"]
NO_TP = "no timepoint"


@dataclass
class View:
    name: str
    dataset: str
    site: str
    kind: str                        # 'compositional' | 'abundance'
    values: pd.DataFrame             # units x features, analysis scale
    raw: pd.DataFrame                # units x features, raw scale
    derived: pd.DataFrame | None     # units x summary indices
    meta: pd.DataFrame               # units: patient, study, group, severity, pasi, timepoint, lesion
    feature_label: str = "feature"
    note: str = ""                   # e.g. samples left out because they fell on the same unit


# =============================================================== units and views
def filter_md(md, f):
    m = md[md["study"].isin(f["studies"]) & md["group"].isin(f["groups"]) & md["severity"].isin(f["severities"])]
    if f.get("timepoints"):
        m = m[m["timepoint"].isna() | m["timepoint"].isin(f["timepoints"])]
    return m


LESION_MODES = {"L1": "most severe lesion only (L1)", "both": "both lesions (L1 and L2)"}


def lesion_tag(md) -> pd.Series:
    """'L1', 'L2', ... for lesional samples (lesion_rank 1 = most severe lesion of the visit), NaN otherwise."""
    rank = pd.to_numeric(md["lesion_rank"], errors="coerce") if "lesion_rank" in md else pd.Series(np.nan, index=md.index)
    les = md["site"].eq("lesional") & rank.notna()
    return pd.Series(np.where(les, "L" + rank.fillna(1).astype(int).astype(str), None), index=md.index, dtype=object)


def unit_key(md):
    """study|subject|timepoint: the key on which modalities are matched; a second lesion adds '|L2'."""
    tp = md["timepoint"].astype(object).where(md["timepoint"].notna(), NO_TP)
    key = md["patient"].astype(str) + "|" + tp.astype(str)
    tag = lesion_tag(md)
    extra = tag.notna() & tag.ne("L1")
    return key.where(~extra, key + "|" + tag.astype(str))


def _unit_meta(md, key):
    """One row per unit. Every unit is one sample, so its values are that sample's own (nothing is averaged)."""
    m = md.assign(lesion=lesion_tag(md).values, timepoint=md["timepoint"].astype(object))
    cols = [c for c in ("patient", "study", "group", "severity", "pasi", "local_severity", "global_severity",
                        "timepoint", "lesion") if c in m]
    meta = m[cols].set_axis(key.values)
    meta = meta[~meta.index.duplicated()]
    meta.index.name = None
    return meta


def build_views(datasets: dict, f: dict) -> dict:
    """One view per dataset and site for the current filters, one row per unit (patient x timepoint, plus a row per
    second lesion with f['lesions'] == 'both'). Each row is one sample: nothing is averaged. If two samples of one
    site still fall on the same unit (two timepoint codes mapped to the same timepoint), the first is used and
    view.notes says so."""
    views = {}
    for ds in datasets.values():
        md = filter_md(ds.md, f)
        if f.get("lesions", "L1") != "both":
            md = E.most_severe_lesion(md)                    # one lesion per patient and visit: the most severe
        present = set(md["site"])
        patient_skin = bool(present & {"lesional", "non-lesional"})
        for site in [x for x in ds.sites if x in present]:
            if site in E.SHARED_HEALTHY_SITES and patient_skin:
                continue                                    # healthy samples join the lesional / NL views
            m = md.loc[E.stratum_samples(md, site)] if site in ("lesional", "non-lesional") else md[md["site"] == site]
            key = unit_key(m)
            dup = key.duplicated()
            name = ds.label if len(ds.sites) == 1 else f"{ds.label} · {ds.site_title(site)}"
            note = ""
            if dup.any():
                note = (f"{name}: {int(dup.sum())} unit(s) have more than one sample (same patient, site and timepoint, "
                        f"e.g. {key[dup].iloc[0]}); the first sample is used, nothing is averaged. Check 'Timepoint "
                        f"labels' on Data setup: two codes of one file are probably mapped to the same timepoint.")
            m, key = m[~dup.values], key[~dup.values]
            one = lambda df: df.loc[m.index].set_axis(key.values)
            views[name] = View(name, ds.name, site, ds.kind, one(ds.values), one(ds.mat),
                               one(ds.derived) if ds.derived is not None else None, _unit_meta(m, key),
                               ds.feature_label, note)
    return views


def units_meta(views: dict) -> pd.DataFrame:
    meta = None
    for v in views.values():
        meta = v.meta if meta is None else meta.combine_first(v.meta)
    return meta


def view_distance(view: View, metric: str = "aitchison") -> pd.DataFrame:
    idx = view.values.index
    if view.kind == "compositional":
        keep = view.raw.columns[(view.raw > 0).any(axis=0)]
        if metric == "braycurtis":
            raw = view.raw[keep].div(view.raw[keep].sum(axis=1), axis=0)
            D = squareform(pdist(raw.values, "braycurtis"))
        else:
            D = squareform(pdist(view.values[keep].values, "euclidean"))
    else:
        D = A.euclidean(view.values)             # z-scored ln values; uses the features both units have
    return pd.DataFrame(D, index=idx, columns=idx)


def unit_coverage(views: dict) -> pd.DataFrame:
    """Units shared by every pair of views (diagonal = units in the view)."""
    names = list(views)
    tab = pd.DataFrame(0, index=names, columns=names)
    for a in names:
        for b in names:
            tab.loc[a, b] = len(views[a].values.index.intersection(views[b].values.index))
    return tab


# =============================================================== ordination and fusion
def ordinate(D: pd.DataFrame, k: int = 3):
    coords, expl = E.pcoa(D.values)
    k = min(k, coords.shape[1])
    return pd.DataFrame(coords[:, :k], index=D.index, columns=[f"PC{i + 1}" for i in range(k)]), expl[:k]


def _rms(D):
    iu = np.triu_indices(len(D), 1)
    v = D.values[iu]
    return float(np.sqrt(np.mean(v ** 2))) if len(v) else 1.0


def fuse(spaces: dict, min_modalities: int | None = None):
    """Combined distance of several spaces (dict name -> distance DataFrame). Returns (D, info).
    min_modalities None or == number of spaces: units present in ALL spaces (exact, recommended).
    Smaller: units with at least that many modalities; each pair of units uses the modalities both have
    (approximate; units without any shared modality with some other unit are dropped)."""
    names = list(spaces)
    K = len(names)
    complete = min_modalities is None or min_modalities >= K
    if complete:
        ids = sorted(set.intersection(*[set(D.index) for D in spaces.values()]))
    else:
        counts = pd.Series(0, index=sorted(set().union(*[set(D.index) for D in spaces.values()])))
        for D in spaces.values():
            counts.loc[D.index] += 1
        ids = list(counts.index[counts >= max(1, min_modalities)])
    num = np.zeros((len(ids), len(ids)))
    cnt = np.zeros_like(num)
    scale = {}
    for n, D in spaces.items():
        sub = D.loc[D.index.isin(ids), D.columns.isin(ids)]
        scale[n] = _rms(sub) or 1.0
        full = (sub / scale[n]).reindex(index=ids, columns=ids).values ** 2
        ok = ~np.isnan(full)
        num[ok] += full[ok]
        cnt[ok] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        fused = pd.DataFrame(np.sqrt(num / cnt), index=ids, columns=ids)
    dropped = 0
    while fused.isna().values.any():                      # pairs without a shared modality
        worst = fused.isna().sum().idxmax()
        fused = fused.drop(index=worst, columns=worst)
        dropped += 1
    out = fused.values.copy()
    np.fill_diagonal(out, 0.0)
    fused = pd.DataFrame(out, index=fused.index, columns=fused.columns)
    info = {"n_units": len(fused), "complete": complete, "dropped_no_overlap": dropped, "scale": scale,
            "n_per_space": {n: int(D.index.isin(fused.index).sum()) for n, D in spaces.items()}}
    return fused, info


def contributions(spaces: dict, fused: pd.DataFrame) -> pd.DataFrame:
    """How strongly each component space is reflected in the combined space (Pearson r of distances)."""
    rows = []
    for n, D in spaces.items():
        ids = fused.index.intersection(D.index)
        iu = np.triu_indices(len(ids), 1)
        r = np.corrcoef(D.loc[ids, ids].values[iu], fused.loc[ids, ids].values[iu])[0, 1] if len(ids) > 2 else np.nan
        rows.append({"space": n, "units": len(ids), "r with combined distances": r})
    return pd.DataFrame(rows)


# =============================================================== permutations
def make_unit_permuter(meta: pd.DataFrame, within_study: bool, rng):
    """Returns f() -> index array p (use X[p]). Whole patients are exchanged, only between patients that
    have the same set of timepoints (and lesions, and the same study if within_study), rows staying aligned by
    timepoint and lesion.
    With one row per patient this is an ordinary (optionally within-study) permutation."""
    tp = meta["timepoint"].astype(object).where(meta["timepoint"].notna(), NO_TP).astype(str)
    if "lesion" in meta:                                     # a second lesion is a slot of its own
        les = meta["lesion"].astype(object)
        tp = tp.where(les.isna() | les.eq("L1"), tp + "|" + les.astype(str))
    tp = tp.values
    pat = meta["patient"].astype(str).values
    study = meta["study"].astype(str).values
    rows = {}
    for i, (p, t) in enumerate(zip(pat, tp)):
        rows.setdefault(p, {})[t] = i
    groups = {}
    for p, d in rows.items():
        key = (tuple(sorted(d)), study[d[next(iter(d))]] if within_study else "")
        groups.setdefault(key, []).append(p)
    plan = [(pats, [[rows[p][t] for t in tps] for p in pats]) for (tps, _), pats in groups.items()]

    def permute():
        idx = np.arange(len(meta))
        for pats, positions in plan:
            order = rng.permutation(len(pats))
            for target, source in zip(positions, [positions[j] for j in order]):
                idx[target] = source
        return idx
    return permute


def _within_patient_permuter(meta, rng):
    pat = pd.factorize(meta["patient"].astype(str))[0]
    groups = [np.where(pat == i)[0] for i in range(pat.max() + 1)]

    def permute():
        idx = np.arange(len(meta))
        for g in groups:
            idx[g] = rng.permutation(g)
        return idx
    return permute


# =============================================================== comparing two spaces
def _standardise(X):
    X = X - X.mean(axis=0)
    n = np.linalg.norm(X)
    return X / n if n > 0 else X


def procrustes_fit(A: np.ndarray, B: np.ndarray):
    """Symmetric Procrustes. Returns (A*, B rotated onto A*, correlation r, m2 = 1 - r^2)."""
    k = max(A.shape[1], B.shape[1])
    A = _standardise(np.pad(A, ((0, 0), (0, k - A.shape[1]))))
    B = _standardise(np.pad(B, ((0, 0), (0, k - B.shape[1]))))
    U, s, Vt = np.linalg.svd(B.T @ A)
    r = float(min(s.sum(), 1.0))
    return A, (B @ U @ Vt) * s.sum(), r, 1 - r ** 2


def protest(Da: pd.DataFrame, Db: pd.DataFrame, meta: pd.DataFrame, k: int = 5, n_perm: int = 999,
            within_study: bool = False, seed: int = 42) -> dict:
    ids = Da.index.intersection(Db.index)
    if len(ids) < 6:
        return {"n": len(ids), "note": "fewer than 6 shared units"}
    A = ordinate(Da.loc[ids, ids], k)[0].values
    B = ordinate(Db.loc[ids, ids], k)[0].values
    r = procrustes_fit(A, B)[2]
    permute = make_unit_permuter(meta.loc[ids], within_study, np.random.default_rng(seed))
    hits = sum(procrustes_fit(A, B[permute()])[2] >= r - 1e-12 for _ in range(n_perm))
    return {"n": len(ids), "axes": int(max(A.shape[1], B.shape[1])), "procrustes r": r, "m2": 1 - r ** 2,
            "p": (hits + 1) / (n_perm + 1)}


def mantel(Da: pd.DataFrame, Db: pd.DataFrame, meta: pd.DataFrame, method: str = "spearman", n_perm: int = 999,
           within_study: bool = False, seed: int = 42) -> dict:
    ids = Da.index.intersection(Db.index)
    if len(ids) < 6:
        return {"n": len(ids), "note": "fewer than 6 shared units"}
    iu = np.triu_indices(len(ids), 1)
    a, B = Da.loc[ids, ids].values[iu], Db.loc[ids, ids].values
    tr = rankdata if method == "spearman" else (lambda x: x)
    a = tr(a)
    a = (a - a.mean()) / a.std()

    def stat(M):
        b = tr(M[iu])
        return float(np.mean(a * (b - b.mean()) / b.std()))
    r = stat(B)
    permute = make_unit_permuter(meta.loc[ids], within_study, np.random.default_rng(seed))
    hits = 0
    for _ in range(n_perm):
        p = permute()
        hits += stat(B[np.ix_(p, p)]) >= r - 1e-12
    return {"n": len(ids), "mantel r": r, "p": (hits + 1) / (n_perm + 1)}


# =============================================================== does a variable explain a space?
def association(D: pd.DataFrame, x: pd.Series, meta: pd.DataFrame, adjust_study: bool = False,
                n_perm: int = 999, seed: int = 42) -> dict:
    """Distance-based regression of a space on one variable (numeric: 1 df, categorical: one df per level).
    Returns R2, pseudo-F, permutation p and the permutation scheme used."""
    x = x.reindex(D.index)
    ok = x.notna().values
    ids = D.index[ok]
    numeric = pd.api.types.is_numeric_dtype(x)
    if len(ids) < 6 or x[ok].nunique() < 2:
        return {"n": int(ok.sum()), "note": "too few units with this variable"}
    m, xv = meta.loc[ids], x[ok]
    G = E.gower(D.loc[ids, ids].values)
    n = len(ids)
    term = ((xv.values.astype(float) - xv.values.astype(float).mean()).reshape(-1, 1) if numeric
            else E._dummies(pd.Categorical(xv.astype(str)).codes, xv.astype(str).nunique()))
    varies_within = bool((xv.groupby(m["patient"].values).nunique() > 1).any())
    cov = [np.ones((n, 1))]
    if adjust_study and m["study"].nunique() > 1:
        cov.append(E.cov_dummies(m, ["study"]))
    if varies_within:                       # repeated units and a within-patient variable: test within patients
        cov.append(E.cov_dummies(m, ["patient"]))
        permute, scheme = _within_patient_permuter(m, np.random.default_rng(seed)), "within patient"
    else:
        permute = make_unit_permuter(m, adjust_study, np.random.default_rng(seed))
        scheme = "whole patients" + (", within study" if adjust_study else "")
    X0 = np.hstack(cov)
    U0, r0 = E._basis(X0)
    ss0, sst = np.sum(U0 * (G @ U0)), np.trace(G)

    def stat(t):
        Uf, rf = E._basis(np.hstack([X0, t]))
        ssf = np.sum(Uf * (G @ Uf))
        dft, dfr = rf - r0, n - rf
        if dft <= 0 or dfr <= 0 or sst - ssf <= 1e-12 * sst:
            return np.nan, np.nan
        return ((ssf - ss0) / dft) / ((sst - ssf) / dfr), (ssf - ss0) / sst
    F, R2 = stat(term)
    if np.isnan(F):
        return {"n": n, "note": "variable is confounded with the covariates"}
    Fp = np.array([stat(term[permute()])[0] for _ in range(n_perm)])
    return {"n": n, "R2": R2, "F": F, "p": (np.nansum(Fp >= F * (1 - 1e-9)) + 1) / (n_perm + 1), "permutation": scheme}


# =============================================================== variables for colouring
def variable_options(view: View) -> dict:
    """{kind: [names]} of the variables one view offers (features sorted by mean level)."""
    opts = {"feature": list(view.raw.mean().sort_values(ascending=False).index)}
    if view.derived is not None and view.derived.shape[1]:
        opts["summary index"] = list(view.derived.columns)
    return opts


def get_variable(spec, views: dict, spaces: dict, meta: pd.DataFrame) -> tuple[pd.Series, str]:
    """spec = (source, kind, name) -> (Series indexed by unit, label).
    source 'Metadata' (kind 'meta'), a view name (kind 'feature' | 'summary index') or a space name (kind 'axis')."""
    source, kind, name = spec
    if kind == "meta":
        return meta[name], name
    if kind == "axis":
        coords, _ = ordinate(spaces[source], 3)
        return coords[name], f"{source}: {name}"
    v = views[source]
    s = v.values[name] if kind == "feature" else v.derived[name]
    if v.site != "lesional":                 # a visit-level value (gut, plasma, non-lesional ...) also colours
        s = _to_lesion_units(s, meta.index)  # the patient's second-lesion (L2) units of that visit
    scale = (" (CLR)" if v.kind == "compositional" else " (ln)") if kind == "feature" else ""
    return s, f"{source}: {name}{scale}"


def _to_lesion_units(s: pd.Series, units) -> pd.Series:
    """Add the units of second lesions ('...|L2') to a Series of visit units, with the value of their visit."""
    extra = [u for u in units if u not in s.index and re.search(r"\|L\d+$", str(u))]
    if not extra:
        return s
    base = [re.sub(r"\|L\d+$", "", str(u)) for u in extra]
    add = pd.Series(s.reindex(base).values, index=extra, name=s.name)
    return pd.concat([s, add.dropna()])


# =============================================================== figures
def _hover(meta, ids, value, label):
    m = meta.reindex(ids)
    val = value.reindex(ids)
    vtxt = val.map(lambda v: "no data" if pd.isna(v) else (f"{v:.3g}" if isinstance(v, (int, float, np.floating)) else str(v)))
    les = (m["lesion"].map(lambda v: f" · lesion {v}" if isinstance(v, str) and v else "") if "lesion" in m
           else pd.Series("", index=m.index))
    return (pd.Series(ids, index=ids).astype(str) + "<br>" + m["group"].astype(str) + " · " + m["severity"].astype(str)
            + les.astype(str)
            + "<br>" + label + ": " + vtxt).values


def _wrap(text, width=30):
    import textwrap
    return "<br>".join(textwrap.wrap(str(text), width)) or str(text)


def spaces_figure(spaces: dict, meta: pd.DataFrame, colour: pd.Series, label: str, axes=(0, 1),
                  shared_only: bool = False, height: int = 520) -> go.Figure:
    """Ordinations of several spaces side by side, all coloured by the same variable. Shape = study.
    Units without a value for the colour variable are drawn as hollow grey markers."""
    names = list(spaces)
    if shared_only and len(names) > 1:
        ids = sorted(set.intersection(*[set(D.index) for D in spaces.values()]))
        spaces = {n: D.loc[ids, ids] for n, D in spaces.items()}
    numeric = pd.api.types.is_numeric_dtype(colour)
    studies = sorted(meta["study"].astype(str).unique())
    sym = {s: STUDY_SYMBOLS[i % len(STUDY_SYMBOLS)] for i, s in enumerate(studies)}
    if not numeric:
        levels = (E.severity_levels(pd.DataFrame({"severity": colour.dropna()})) if label == "severity"
                  else sorted(colour.dropna().astype(str).unique()))
        cmap = E.severity_colors(levels) if label == "severity" else P.categorical_colors(levels)
    ok_spaces = {n: D for n, D in spaces.items() if len(D) >= 3}
    titles = [f"{n} (n={len(D)})" for n, D in ok_spaces.items()]
    fig = make_subplots(rows=1, cols=max(len(ok_spaces), 1), subplot_titles=titles, horizontal_spacing=0.06)
    a, b = axes
    shown = set()
    for c, (n, D) in enumerate(ok_spaces.items(), start=1):
        coords, expl = ordinate(D, max(axes) + 1)
        if coords.shape[1] <= max(axes):
            continue
        ids = coords.index
        col = colour.reindex(ids)
        st = meta["study"].reindex(ids).astype(str)
        hov = _hover(meta, ids, colour, label)
        miss = col.isna().values
        healthy = np.zeros(len(ids), bool)
        if label in P.SCORE_LABELS:                       # severity scores: healthy volunteers are green
            healthy = (meta["group"].reindex(ids) == E.HEALTHY).values
            miss = miss | healthy                         # drawn green, not on the colour scale
        for s in studies:
            in_s = (st == s).values
            hh = in_s & healthy
            if hh.any():
                fig.add_trace(go.Scatter(x=coords.values[hh, a], y=coords.values[hh, b], mode="markers", text=hov[hh],
                                         hoverinfo="text", name="healthy", legendgroup="healthy",
                                         showlegend="healthy" not in shown,
                                         marker=dict(symbol=sym[s], color=P.HEALTHY_COLOR, size=10,
                                                     line=dict(width=0.7, color="#222"))), row=1, col=c)
                shown.add("healthy")
            mm = in_s & miss & ~healthy
            if mm.any():
                fig.add_trace(go.Scatter(x=coords.values[mm, a], y=coords.values[mm, b], mode="markers", text=hov[mm],
                                         hoverinfo="text", name="no data", legendgroup="nodata",
                                         showlegend="nodata" not in shown,
                                         marker=dict(symbol=sym[s] + "-open", color="#9a9a9a", size=9)), row=1, col=c)
                shown.add("nodata")
            if numeric:
                mk = in_s & ~miss
                if mk.any():
                    fig.add_trace(go.Scatter(x=coords.values[mk, a], y=coords.values[mk, b], mode="markers",
                                             text=hov[mk], hoverinfo="text", showlegend=False,
                                             marker=dict(symbol=sym[s], size=10, color=col.values[mk].astype(float),
                                                         coloraxis="coloraxis", line=dict(width=0.7, color="#222"))),
                                  row=1, col=c)
            else:
                for lv in levels:
                    mk = in_s & (col.astype(str) == lv).values & ~miss
                    if mk.any():
                        fig.add_trace(go.Scatter(x=coords.values[mk, a], y=coords.values[mk, b], mode="markers",
                                                 text=hov[mk], hoverinfo="text", name=lv, legendgroup=lv,
                                                 showlegend=lv not in shown, legendrank=10 + levels.index(lv),
                                                 marker=dict(symbol=sym[s], size=10, color=cmap[lv],
                                                             line=dict(width=0.7, color="#222"))), row=1, col=c)
                        shown.add(lv)
        fig.update_xaxes(title_text=f"PCo{a + 1} ({expl[a]:.1%})", row=1, col=c)
        fig.update_yaxes(title_text=f"PCo{b + 1} ({expl[b]:.1%})", row=1, col=c)
    for s in studies:                                     # shape legend
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"study {s}", legendgroup="study",
                                 legendrank=900,
                                 marker=dict(symbol=sym[s], color="#555", size=10)), row=1, col=1)
    fig.update_layout(template=P.TEMPLATE, height=height, margin=dict(t=60),
                      legend=dict(title=f"colour = {_wrap(label)}<br>shape = study", font=dict(size=10)))
    if numeric:
        scale = P.LOCAL_SCALE if label in P.SCORE_LABELS else "Viridis"      # severity scores: blue -> red
        fig.update_layout(coloraxis=dict(colorscale=scale,
                                         colorbar=dict(title=dict(text=_wrap(label, 24), font=dict(size=10)),
                                                       len=0.55, y=0.28)),
                          legend=dict(y=1, yanchor="top"))
    return fig


def procrustes_figure(Da, Db, name_a, name_b, meta, colour, label, k=5) -> go.Figure:
    """Space B rotated onto space A; a line joins the two positions of every unit (short = they agree)."""
    ids = Da.index.intersection(Db.index)
    A = ordinate(Da.loc[ids, ids], k)[0].values
    B = ordinate(Db.loc[ids, ids], k)[0].values
    A, Br, r, _ = procrustes_fit(A, B)
    fig = go.Figure()
    xs, ys = [], []
    for i in range(len(ids)):
        xs += [A[i, 0], Br[i, 0], None]
        ys += [A[i, 1], Br[i, 1], None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#bbbbbb", width=1), hoverinfo="skip",
                             showlegend=False))
    hov = _hover(meta, ids, colour, label)
    for pts, nm, symbol in ((A, name_a, "circle"), (Br, name_b, "diamond-open")):
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", name=nm, text=hov, hoverinfo="text",
                                 marker=dict(symbol=symbol, size=9, color="#2b6cb0" if symbol == "circle" else "#c0392b")))
    fig.update_layout(template=P.TEMPLATE, height=520, xaxis_title="Procrustes axis 1", yaxis_title="Procrustes axis 2",
                      title=f"{name_b} rotated onto {name_a} (n={len(ids)}, Procrustes r = {r:.2f})")
    return fig
