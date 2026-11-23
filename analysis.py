"""
analysis.py -- statistics and figures for concentration-type modalities (ceramides, energy metabolites).

It reuses the permutation engine of the microbiome pages (skin_engine), so every modality gets the same
design: blocks (study / disease group), strata (site), one unit per patient, paired timepoint tests,
PERMANOVA + PERMDISP and multiple-testing correction per family.

Differences from the microbiome analysis
  * values are ln(concentration); a difference of means is a log fold change (reported as log2FC)
  * zero / missing values are NaN unless imputation is on: a feature is then tested on the patients that
    have a value for it (the 'n' column shows how many)
  * whole-profile distance = Euclidean distance on z-scored ln values over the features both samples have,
    scaled to the full number of features (no value is filled in)
  * the 'alpha' table holds the modality's summary indices (class fractions, carbon number, ratios)
"""
from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import rankdata

import datasets as DS
import skin_engine as E
import skin_plots as P


def zscore(values: pd.DataFrame) -> pd.DataFrame:
    sd = values.std(ddof=0)
    keep = sd.index[sd > 0]
    return (values[keep] - values[keep].mean()) / sd[keep]


def euclidean(values: pd.DataFrame, min_frac: float = 0.5) -> np.ndarray:
    """Distance between rows on z-scored values; features measured in < min_frac of the rows are left out."""
    z = zscore(values)
    z = z.loc[:, z.notna().mean() >= min_frac]
    D = DS.nan_euclidean(z.values) if z.shape[1] else np.zeros((len(values), len(values)))
    if np.isnan(D).any():                                   # two samples without a feature in common
        D = np.where(np.isnan(D), np.nanmax(D), D)
    return D


def _tests(Y, meta, factor, levels, ordered, block, S, seeds, with_rel):
    """engine.feature_tests on every group of columns that share the same missing-value pattern."""
    rows, groups, present = [], {}, Y.notna()
    for c in Y.columns:
        groups.setdefault(present[c].values.tobytes(), []).append(c)
    for cols in groups.values():
        ok = present[cols[0]].values
        Ys, ms = Y.loc[ok, cols], meta.loc[ok]
        lv = [l for l in levels if l in set(ms[factor])]
        if ok.sum() < 4 or len(lv) < 2:
            continue
        rows += E.feature_tests(Ys, ms, factor, lv, ordered, block.cov_candidates, block.perm_block, S, seeds,
                                rel=np.exp(Ys) if with_rel else None)
    return rows


def _analyze_block(block, md, values, mat, derived, S, factor, levels_all, ordered, seeds):
    mb = md.loc[block.names].copy()
    V, M = values.loc[block.names], mat.loc[block.names]
    idx = None
    if derived is not None and derived.shape[1]:
        idx = derived.loc[block.names]
        idx = idx.loc[:, idx.nunique() > 1]
        idx = idx if idx.shape[1] else None
    dists = {"euclidean": euclidean(V)}
    out = {k: [] for k in ("alpha", "da", "beta", "wdist", "counts")}
    tested = set()
    for st in E.strata_for(set(mb["site"])):
        names = E.stratum_samples(mb, st, factor)
        if len(names) == 0:
            continue
        feats = list(M.columns[(M.loc[names] > 0).mean() >= S.min_prev])
        Yv, meta = E.unit_frame(mb, V[feats] if feats else V.iloc[:, :1], st, factor=factor)
        levels = [l for l in levels_all if l in set(meta[factor])]
        for lv in levels:
            out["counts"].append(dict(stratum=st, level=lv, n_units=int((meta[factor] == lv).sum()),
                                      n_patients=int(meta.loc[meta[factor] == lv, "patient"].nunique())))
        if len(levels) < 2:
            continue
        if idx is not None:
            Yi, _ = E.unit_frame(mb, idx, st, factor=factor)
            for r in _tests(Yi, meta, factor, levels, ordered, block, S, seeds, False):
                out["alpha"].append({**r, "stratum": st})
        if feats:
            tested.update(feats)
            for r in _tests(Yv, meta, factor, levels, ordered, block, S, seeds, True):
                out["da"].append({**r, "stratum": st})
        if factor == "timepoint":
            Yb, meta_b = E.unit_frame(mb, V, st, factor=factor)
            out["beta"] += E.beta_tests({"euclidean": euclidean(Yb)}, mb, st, factor, levels, ordered,
                                        block.cov_candidates, block.perm_block, S, seeds, unit_meta=meta_b)
        else:
            out["beta"] += E.beta_tests(dists, mb, st, factor, levels, ordered, block.cov_candidates,
                                        block.perm_block, S, seeds)
    if {"lesional", "non-lesional"} <= set(mb["site"]):
        wp = E.within_patient_frame(dists, mb, factor=factor)
        if len(wp):
            meta = wp.drop(columns=list(dists)).reset_index(drop=True)
            levels = [l for l in levels_all if l in set(meta[factor])]
            for r in E.feature_tests(wp[list(dists)].reset_index(drop=True), meta, factor, levels, ordered,
                                     block.cov_candidates, block.perm_block, S, seeds):
                out["wdist"].append({**r, "stratum": "within-patient L-NL"})
    frames = {}
    for k, rows in out.items():
        df = pd.DataFrame(rows)
        if len(df):
            df.insert(0, "block", block.label)
        frames[k] = df
    frames["taxa"] = sorted(tested)
    return frames


def run_abundance_analysis(md, values, mat, S, factor="severity", levels_all=None, ordered=None, derived=None,
                           progress=None):
    """Same contract as skin_engine.run_analysis: returns (tables, blocks)."""
    fm = E.filter_md(md, S)
    if levels_all is None:
        levels_all = (E.severity_levels(fm) if factor == "severity" else E.timepoint_levels(fm)
                      if factor == "timepoint" else sorted(fm[factor].dropna().astype(str).unique()))
    if ordered is None:
        ordered = factor == "severity"
    notes = []
    if (factor != "timepoint" and S.timepoint_mode == "average"
            and (fm.groupby(["patient", "site"])["timepoint"].nunique() > 1).any()):
        notes.append("Timepoints are averaged: each unit is the patient mean across the selected timepoints.")
    if factor != "timepoint" and S.timepoint_mode == "separate" and E.timeless_healthy(fm).any() \
            and fm.loc[~E.timeless_healthy(fm), "timepoint"].nunique() > 1:
        notes.append("Healthy volunteers sampled at one timepoint only are the reference at every timepoint.")
    blocks = E.make_blocks(md, S, factor)
    seeds = itertools.count(S.seed)
    frames = []
    for i, b in enumerate(blocks):
        if progress:
            progress(i / max(len(blocks), 1), f"Analysing block {i + 1}/{len(blocks)}: {b.label}")
        frames.append(_analyze_block(b, md, values, mat, derived, S, factor, levels_all, ordered, seeds))
    if progress:
        progress(1.0, "Done")
    tabs = E.finalize(frames, S)
    tabs["notes"] = notes
    return tabs, blocks


# =============================================================== site comparison (paired within patient)
def run_site_analysis(md, values, mat, S, kind, derived=None, progress=None):
    """Paired comparison between sites (lesional vs non-lesional, nostril, blister vs plasma ...) within a
    patient, for both microbiome (kind='compositional') and concentration data (kind='abundance').
    One block per study / disease group (see Settings) and timepoint; inside a block a pair is the same
    patient's samples of two sites. Patient is a covariate and site labels are permuted within patient.
    Returns (tables, blocks) like the other analyses; the contrast column reads e.g. 'lesional vs non-lesional'."""
    blocks = []
    for b in E.make_blocks(md, S, "site"):                 # pairs are formed within a timepoint, never across
        mb = md.loc[b.names]
        tps = [t for t in E.timepoint_levels(mb) if (mb["timepoint"] == t).sum() >= 4]
        if len(tps) <= 1 and not mb["timepoint"].isna().any():
            blocks.append(b)
            continue
        for t in tps:
            blocks.append(E.Block(f"{b.label} | {t}", list(mb.index[(mb["timepoint"] == t).values]),
                                  b.cov_candidates, b.perm_block))
        if mb["timepoint"].isna().sum() >= 4:
            blocks.append(E.Block(f"{b.label} | no timepoint", list(mb.index[mb["timepoint"].isna().values]),
                                  b.cov_candidates, b.perm_block))
    seeds = itertools.count(S.seed)
    frames = []
    for i, block in enumerate(blocks):
        if progress:
            progress(i / max(len(blocks), 1), f"Analysing block {i + 1}/{len(blocks)}: {block.label}")
        mb = md.loc[block.names]
        keys = [mb["patient"].values, mb["site"].values]
        unit = lambda df: df.loc[mb.index].groupby(keys, sort=True).mean()
        V, M = unit(values), unit(mat)
        meta = (mb.groupby(["patient", "site"], sort=True)
                .agg(study=("study", "first"), group=("group", "first"), severity=("severity", "first"),
                     skin_type=("skin_type", "first")).reset_index())
        meta["timepoint"] = meta["site"]                      # the engine's paired design keys on this column
        V, M = V.reset_index(drop=True), M.reset_index(drop=True)
        levels = [s for s in P.site_order(meta["site"]) if (meta["site"] == s).sum() >= 2]
        out = {k: [] for k in ("alpha", "da", "beta", "wdist", "counts")}
        for lv in levels:
            out["counts"].append(dict(stratum="within patient", level=lv, n_units=int((meta["site"] == lv).sum()),
                                      n_patients=int(meta.loc[meta["site"] == lv, "patient"].nunique())))
        feats = []
        if len(levels) >= 2:
            cov, pb = block.cov_candidates, block.perm_block
            if derived is not None and derived.shape[1]:
                Yi = unit(derived).reset_index(drop=True)
                Yi = Yi.loc[:, Yi.nunique() > 1]
                out["alpha"] += _tests(Yi, meta, "timepoint", levels, False, block, S, seeds, False)
            if kind == "compositional":
                prev, mean = (mat.loc[mb.index] > 0).mean(), mat.loc[mb.index].mean()
                feats = [t for t in mat.columns[(prev >= S.min_prev) & (mean >= S.min_abund)] if t not in E.EXCLUDE_TAXA]
                if feats:
                    out["da"] += E.feature_tests(V[feats], meta, "timepoint", levels, False, cov, pb, S, seeds, rel=M[feats])
                dists = E.distance_matrices(M.div(M.sum(axis=1), axis=0), V)
            else:
                feats = list(mat.columns[(mat.loc[mb.index] > 0).mean() >= S.min_prev])
                if feats:
                    out["da"] += _tests(V[feats], meta, "timepoint", levels, False, block, S, seeds, True)
                dists = {"euclidean": euclidean(V)}
            out["beta"] += E.beta_tests(dists, None, "within patient", "timepoint", levels, False, cov, pb, S, seeds,
                                        unit_meta=meta)
        fr = {}
        for k, rows in out.items():
            df = pd.DataFrame(rows)
            if len(df):
                df["stratum"] = "within patient"
                df.insert(0, "block", block.label)
            fr[k] = df
        fr["taxa"] = sorted(feats)
        frames.append(fr)
    if progress:
        progress(1.0, "Done")
    tabs = E.finalize(frames, S)
    tabs["notes"] = []
    return tabs, blocks


SITE_METHODS = """Sites were compared within patients: within each block (study, disease group and timepoint) a pair is
the same patient's samples of two sites, patient is a fixed-effect covariate and site labels are permuted within
patient. Only patients with both sites of a pair contribute to that pair. Effect sizes are paired (mean difference
as log2 fold change, matched-pairs rank-biserial correlation). Whole-profile differences were tested with PERMANOVA
using the same within-patient permutations.
"""


def methods_text(S, factor, feature_label="feature", imputation="none"):
    """Methods paragraph for concentration-type modalities (same design as the microbiome pages)."""
    missing = {"none": f"Values that were zero or missing were treated as not measured and left out: every "
                       f"{feature_label} was tested on the patients with a value for it, without imputation.",
               "half-min": f"Values that were zero or missing were treated as below the detection limit and replaced "
                           f"by half the smallest measured value of that {feature_label}.",
               "knn": f"Values that were zero or missing were imputed with the mean of the five most similar samples "
                      f"of the same specimen type (z-scored ln scale)."}[imputation]
    parts = E.methods_text(S, factor=factor).split("\n\n")
    parts[1] = (f"Concentrations were natural-log transformed. {missing} A {feature_label} was tested within a "
                f"stratum when measured above zero in >= {S.min_prev:.0%} of that stratum's samples. Differences are "
                f"reported as log2 fold changes. Whole-profile comparisons used Euclidean distances on z-scored ln "
                f"values, computed over the features both samples have and scaled to the full number of features. "
                f"Summary indices (class fractions, carbon number, ratios) were tested like single features.")
    return ("\n\n".join(parts).replace("Effect sizes: Cliff's delta, log2 fold change (CLR).",
                                        "Effect sizes: Cliff's delta, log2 fold change.")
            .replace("Beta diversity: Bray-Curtis and Aitchison distances; PERMANOVA", "Whole profile: PERMANOVA"))


# =============================================================== local severity (lesional samples only)
LS_BETWEEN = "lesional · between patients"
LS_WITHIN = "lesional · within patient"


def _by_pattern(Y):
    """Groups of columns of Y that share the same missing-value pattern -> [(row mask, columns)]."""
    groups, present = {}, Y.notna()
    for c in Y.columns:
        groups.setdefault(present[c].values.tobytes(), []).append(c)
    return [(present[cols[0]].values, cols) for cols in groups.values()]


def _ls_association(Y, meta, block, S, seeds, min_n=5):
    """Between patients (one unit per patient): Spearman-type association of every column of Y with local
    severity. Permutation F test on ranks (= Spearman without covariates), adjusted for the block covariates.
    effect = Spearman rho; diff = least-squares slope of the value per unit of local severity."""
    rows = []
    for ok, cols in _by_pattern(Y):
        ms = meta.loc[ok]
        x = ms["local_severity"].astype(float).values
        if ok.sum() < min_n or np.unique(x).size < 2:
            continue
        Ys = Y.loc[ok, cols].values.astype(float)
        covs = [c for c in block.cov_candidates if c in ms.columns and ms[c].fillna("n/a").nunique() > 1]
        cov = E.cov_dummies(ms, covs)
        blocks = ms[block.perm_block].values if block.perm_block in covs else None
        R, rx = rankdata(Ys, axis=0), rankdata(x)
        r = E.perm_test(R, rx, None, cov, "free", None, blocks, S.n_perm, next(seeds), "feat")
        Rc, xc = R - R.mean(0), rx - rx.mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = (Rc.T @ xc) / (np.sqrt((Rc ** 2).sum(0)) * np.sqrt((xc ** 2).sum()))
            slope = ((Ys - Ys.mean(0)).T @ (x - x.mean())) / ((x - x.mean()) ** 2).sum()
        for j, f in enumerate(cols):
            rows.append(dict(feature=f, test="association", contrast="local severity (per unit)", lo=None, hi=None,
                             F=r["F"][j], effect=rho[j], p=r["p"][j], n=int(ok.sum()), covariates=",".join(covs),
                             diff=slope[j], stratum=LS_BETWEEN))
    return rows


def _signed_rank_test(d, n_perm, seed):
    """Wilcoxon signed-rank statistic per column of d (pairs x features, no NaN); two-sided p from sign flips
    (all 2^n sign patterns when that is at most n_perm + 1, else n_perm random ones)."""
    n = d.shape[0]
    sr = np.sign(d) * rankdata(np.abs(d), axis=0)
    T = sr.sum(0)
    if 2 ** n <= n_perm + 1:
        signs = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
        Tp = signs @ sr
        p = (np.abs(Tp) >= np.abs(T) * (1 - 1e-9)).mean(0)
    else:
        signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n_perm, n))
        Tp = signs @ sr
        p = ((np.abs(Tp) >= np.abs(T) * (1 - 1e-9)).sum(0) + 1) / (n_perm + 1)
    return T, np.where(np.abs(sr).sum(0) > 0, p, np.nan)


def _lesion_pairs(lesional):
    """Patients with two (or more) lesions at one timepoint that differ in local severity: the most and the
    least severe lesion form a pair. Returns DataFrame(patient, timepoint, hi, lo, delta) with sample ids."""
    rows = []
    tp = lesional["timepoint"].astype(object).where(lesional["timepoint"].notna(), "")
    for (pat, t), g in lesional.assign(_tp=tp.values).groupby(["patient", "_tp"], sort=True):
        g = g.sort_values("local_severity")
        if len(g) >= 2 and g["local_severity"].iloc[-1] > g["local_severity"].iloc[0]:
            rows.append(dict(patient=pat, timepoint=t, hi=g.index[-1], lo=g.index[0],
                             delta=float(g["local_severity"].iloc[-1] - g["local_severity"].iloc[0])))
    return pd.DataFrame(rows, columns=["patient", "timepoint", "hi", "lo", "delta"])


def _ls_within(Y, pairs, S, seeds, min_pairs=4):
    """Within patient: more severe minus less severe lesion; a patient with pairs at several timepoints
    contributes the mean difference. Signed-rank sign-flip test per column."""
    rows = []
    d_all = pd.DataFrame(Y.loc[pairs["hi"]].values - Y.loc[pairs["lo"]].values, columns=Y.columns)
    d_all = d_all.groupby(pairs["patient"].values).mean()        # one difference per patient
    for ok, cols in _by_pattern(d_all):
        if ok.sum() < min_pairs:
            continue
        d = d_all.loc[ok, cols].values.astype(float)
        T, p = _signed_rank_test(d, S.n_perm, next(seeds))
        eff = E._paired_effect(d)
        for j, f in enumerate(cols):
            rows.append(dict(feature=f, test="association", contrast="more vs less severe lesion", lo=None,
                             hi=None, F=T[j], effect=eff[j], p=p[j], n=int(ok.sum()), n_pairs=int(ok.sum()),
                             covariates="patient", diff=d[:, j].mean(), stratum=LS_WITHIN))
    return rows


def run_local_severity_analysis(md, values, mat, S, kind, derived=None, pseudo=None, progress=None):
    """Association of the lesional microbiome / ceramides with the local severity of the lesion.
    md, values, mat: ALL samples (unfiltered): S is applied here, and the within-patient test needs every
    lesion, whatever S.lesions says. kind: 'compositional' (values = CLR, mat = relative abundance) or
    'abundance' (values = ln concentration with NaN, mat = concentration).
    Two designs per block (study / disease group / timepoint, as for the other comparisons):
      between patients -- one unit per patient = its most severe lesion; rank association with local severity
      within patient   -- patients with two lesions of different local severity: more vs less severe lesion
    Returns (tables, blocks) like the other analyses."""
    full = E.filter_md(md, replace(S, lesions="both"))
    full = full[(full["site"] == "lesional") & pd.to_numeric(full["local_severity"], errors="coerce").notna()]
    full = full.assign(local_severity=pd.to_numeric(full["local_severity"], errors="coerce"))
    blocks = [b for b in E.make_blocks(md, S, "local_severity") if len(full.index.intersection(b.names))]
    seeds = itertools.count(S.seed)
    frames, notes = [], []
    if derived is not None:
        derived = derived.loc[:, derived.nunique() > 1]
    for i, block in enumerate(blocks):
        if progress:
            progress(i / max(len(blocks), 1), f"Analysing block {i + 1}/{len(blocks)}: {block.label}")
        out = {k: [] for k in ("alpha", "da", "beta", "wdist", "counts")}
        mb = full.loc[full.index.intersection(block.names)]       # most severe lesion per patient and timepoint
        if kind == "compositional":
            prev, mean = (mat.loc[mb.index] > 0).mean(), mat.loc[mb.index].mean()
            feats = [t for t in mat.columns[(prev >= S.min_prev) & (mean >= S.min_abund)] if t not in E.EXCLUDE_TAXA]
        else:
            feats = list(mat.columns[(mat.loc[mb.index] > 0).mean() >= S.min_prev])
        # ---- between patients
        _, meta = E.unit_frame(mb, values.loc[mb.index].iloc[:, :1], "lesional")
        if meta is not None:
            out["counts"].append(dict(stratum=LS_BETWEEN, level="patients", n_units=len(meta),
                                      n_patients=int(meta["patient"].nunique())))
            if meta["local_severity"].nunique() >= 2:
                if derived is not None and derived.shape[1]:
                    Yi, _ = E.unit_frame(mb, derived.loc[mb.index], "lesional")
                    out["alpha"] += _ls_association(Yi, meta, block, S, seeds)
                if feats:
                    Yv, _ = E.unit_frame(mb, values.loc[mb.index, feats], "lesional")
                    out["da"] += _ls_association(Yv, meta, block, S, seeds)
                if kind == "compositional":
                    Yr, _ = E.unit_frame(mb, mat.loc[mb.index], "lesional")
                    Yr = Yr.loc[:, (Yr > 0).any(axis=0)]
                    Yr = Yr.div(Yr.sum(axis=1), axis=0)
                    dists = E.distance_matrices(Yr, E.clr(Yr, pseudo))
                else:
                    Yv, _ = E.unit_frame(mb, values.loc[mb.index], "lesional")
                    dists = {"euclidean": euclidean(Yv)}
                x = meta["local_severity"].astype(float).values
                covs = [c for c in block.cov_candidates if c in meta.columns and meta[c].fillna("n/a").nunique() > 1]
                pb = meta[block.perm_block].values if block.perm_block in covs else None
                if len(meta) >= 5:
                    for metric, D in dists.items():
                        r = E.perm_test(D, x, None, E.cov_dummies(meta, covs), "free", None, pb, S.n_perm,
                                        next(seeds), "dist")
                        out["beta"].append(dict(test="PERMANOVA", kind="local_between", metric=metric,
                                                stratum=LS_BETWEEN, contrast="local severity (per unit)", lo=None,
                                                hi=None, n=len(meta), covariates=",".join(covs), n_pairs=np.nan, **r))
        # ---- within patient (two lesions)
        keys = set(zip(mb["patient"], mb["timepoint"].astype(object).where(mb["timepoint"].notna(), "")))
        tp_all = full["timepoint"].astype(object).where(full["timepoint"].notna(), "")
        lesional = full[[k in keys for k in zip(full["patient"], tp_all)]]
        pairs = _lesion_pairs(lesional)
        if len(pairs):
            out["counts"].append(dict(stratum=LS_WITHIN, level="patients with two lesions",
                                      n_units=2 * len(pairs), n_patients=int(pairs["patient"].nunique())))
        if pairs["patient"].nunique() >= 4:
            if derived is not None and derived.shape[1]:
                out["alpha"] += _ls_within(derived, pairs, S, seeds)
            if feats:
                out["da"] += _ls_within(values[feats], pairs, S, seeds)
            ids = list(pairs["hi"]) + list(pairs["lo"])
            codes = np.r_[np.ones(len(pairs)), np.zeros(len(pairs))].astype(int)
            strata = np.r_[np.arange(len(pairs)), np.arange(len(pairs))]
            if kind == "compositional":
                M = mat.loc[ids]
                M = M.loc[:, (M > 0).any(axis=0)]
                dists = E.distance_matrices(M.div(M.sum(axis=1), axis=0), values.loc[ids])
            else:
                dists = {"euclidean": euclidean(values.loc[ids])}
            cov = E.cov_dummies(pd.DataFrame({"pair": strata}), ["pair"])
            for metric, D in dists.items():
                r = E.perm_test(D, codes, 2, cov, "within", strata, None, S.n_perm, next(seeds), "dist")
                out["beta"].append(dict(test="PERMANOVA", kind="local_within", metric=metric, stratum=LS_WITHIN,
                                        contrast="more vs less severe lesion", lo=None, hi=None, n=len(ids),
                                        covariates="pair", n_pairs=len(pairs), **r))
        elif len(pairs):
            notes.append(f"{block.label}: only {pairs['patient'].nunique()} patient(s) with two lesions of "
                         f"different local severity; the within-patient test needs at least 4.")
        fr = {}
        for k, rows in out.items():
            df = pd.DataFrame(rows)
            if len(df):
                df.insert(0, "block", block.label)
            fr[k] = df
        fr["taxa"] = sorted(feats)
        frames.append(fr)
    if progress:
        progress(1.0, "Done")
    if not blocks:
        notes.append("No lesional samples with a local severity in the selection.")
    tabs = E.finalize(frames, S)
    tabs["notes"] = notes
    return tabs, blocks


def local_severity_methods(S, kind, feature_label="taxon"):
    values = ("centred log-ratio (CLR) abundances" if kind == "compositional"
              else "natural-log concentrations (a missing or zero value leaves that sample out for that feature)")
    whole = ("Bray-Curtis and Aitchison distances" if kind == "compositional"
             else "Euclidean distances on z-scored ln values")
    return f"""LOCAL SEVERITY (auto-generated -- edit to match your final analysis)

Only lesional samples with a recorded local severity were used; local severity was analysed as a number.
Analyses were run within the same blocks as the other comparisons (study, disease group and timepoint, as
selected in the sidebar).

Between patients: one unit per patient, its most severe lesion (per timepoint, or the patient mean when timepoints are averaged). Every
{feature_label} ({values}) and summary index was tested for a monotonic association with local severity:
permutation F test on ranks ({S.n_perm} permutations; equivalent to a Spearman correlation without covariates),
adjusted for the block covariates. Effect sizes: Spearman rho and the least-squares slope per unit of local
severity (log2 scale in the 'log2fc' column). Whole profile: PERMANOVA with local severity as a continuous term
({whole}).

Within patient: patients with two lesions of different local severity at the same timepoint. The more severe
minus the less severe lesion was tested with a Wilcoxon signed-rank statistic and sign-flip permutations (exact
when there are few pairs); a patient with pairs at several timepoints contributes the mean difference. Effect
size: matched-pairs rank-biserial correlation; 'log2fc' = mean difference on the log2 scale. Whole profile:
PERMANOVA with lesion (more vs less severe) as a term and permutations within patient. At least 4 patients are
needed.

Multiple testing: {S.correction.upper()} within families (features within block x design; whole-profile tests
within block x design x metric); 'q_global' across blocks. Studies may score local severity differently: run
them separately, or adjust for study. Results are exploratory.
"""


# =============================================================== figures
def profile_heatmap(values, md, factor, levels, max_features=60):
    """Mean z-score of every feature per (site, factor level): a quick overview of what changes where."""
    z = zscore(values)
    if z.shape[1] > max_features:
        z = z[z.var().nlargest(max_features).index]
    key = md["site"].astype(str) + " · " + md[factor].astype(str)
    order = [f"{s} · {l}" for s in P.site_order(md["site"]) for l in levels if f"{s} · {l}" in set(key)]
    means = z.groupby(key.values).mean().reindex(order)
    fig = go.Figure(go.Heatmap(z=means.T.values, x=order, y=list(means.columns), zmid=0, colorscale="RdBu_r",
                               colorbar=dict(title="mean z-score")))
    fig.update_layout(template=P.TEMPLATE, height=max(380, 15 * means.shape[1] + 160), xaxis=dict(tickangle=-40),
                      yaxis=dict(autorange="reversed"), margin=dict(l=120))
    return fig


def class_composition_figure(mat, md, info, levels, by="cer_class"):
    """Mean share (%) of each ceramide class (or other annotation) per severity and site."""
    cls = info.reindex(mat.columns)[by].astype(str)
    share = mat.T.groupby(cls.values).sum().T
    share = 100 * share.div(share.sum(axis=1), axis=0)
    long = (share.join(md[["severity", "site"]]).groupby(["severity", "site"]).mean().reset_index()
            .melt(id_vars=["severity", "site"], var_name="class", value_name="share"))
    fig = px.bar(long, x="site", y="share", color="class", facet_col="severity", template=P.TEMPLATE, height=430,
                 category_orders={"severity": [l for l in levels if l in set(long["severity"])],
                                  "site": P.site_order(long["site"])},
                 color_discrete_sequence=px.colors.qualitative.Safe)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_yaxes(title_text="mean share of total (%)", matches=None, range=[0, 100])
    fig.update_xaxes(title_text="")
    return fig


def chain_length_figure(mat, md, info, levels, sev_colors):
    """Mean share of the non-EO ceramides per carbon number, one line per severity, one panel per site."""
    info = info.reindex(mat.columns)
    sel = ((info["fa_code"] != "EO") & info["carbons"].notna()).values
    if sel.sum() < 3:
        return None
    by_c = mat.loc[:, sel].T.groupby(info.loc[sel, "carbons"].astype(int).values).sum().T
    by_c = 100 * by_c.div(by_c.sum(axis=1), axis=0)
    sites = P.site_order(md["site"])
    fig = P.make_subplots(rows=1, cols=len(sites), shared_yaxes=True, subplot_titles=sites)
    for c, site in enumerate(sites):
        for lv in levels:
            rows = md.index[(md["site"] == site) & (md["severity"] == lv)]
            if len(rows) == 0:
                continue
            m = by_c.loc[rows].mean()
            fig.add_trace(go.Scatter(x=m.index, y=m.values, mode="lines+markers", name=lv, legendgroup=lv,
                                     showlegend=(c == 0),
                                     line=dict(color=sev_colors.get(lv, "#888"),
                                               dash="dash" if E.severity_scale(lv) == 0 else "solid")), row=1, col=c + 1)
        fig.update_xaxes(title_text="carbon number (as in the ceramide name)", row=1, col=c + 1)
    fig.update_yaxes(title_text="mean share of non-EO ceramides (%)", row=1, col=1)
    fig.update_layout(template=P.TEMPLATE, height=420)
    return fig
