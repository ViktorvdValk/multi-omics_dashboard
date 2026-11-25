"""
skin_engine.py -- data loading, permutation statistics and analysis orchestration
for the multi-study skin microbiome dashboard (no Streamlit/Plotly in here).

DESIGN
  * Every sample has: study, patient, site (non-lesional / lesional / nostril),
    group (healthy / MF / FMF / psoriasis ...), severity (ordinal label; 'healthy' for
    controls), skin_type (moist / sebaceous / dry, may be missing for nostril) and,
    when available, timepoint (e.g. pre / post treatment).
  * Analyses run per BLOCK. A block is a subset of samples that is analysed together;
    blocks come from the "separate" choices (study, disease group, skin type).
    "Adjust" choices become covariates, with permutations restricted within study.
  * Strata inside a block: non-lesional, lesional, pooled (L+NL, patient-level mean,
    patients with both only), nostril and gut (faecal; its own stratum, never pooled with
    skin). One unit = one patient (or one patient-timepoint).
  * Gut and skin communities share almost no taxa, so (a) the taxon prevalence/abundance filter
    is applied PER STRATUM and (b) CLR is computed per compartment (gut vs skin/nostril) over
    the taxa detected in that compartment (see clr_by_compartment).
  * Univariate tests (alpha diversity, CLR taxa, within-patient distance) are rank-based
    permutation tests (rank-ANOVA F; equals Mann-Whitney / Kruskal-Wallis / Spearman
    without covariates, but can adjust for covariates).
  * Beta diversity: PERMANOVA (marginal, adonis2-style) + PERMDISP, Bray-Curtis and
    Aitchison. Permutations are restricted: whole patients are permuted, within study
    when study is a covariate; lesional vs non-lesional is permuted within patient.
  * Timepoint comparisons (pre vs post) are PAIRED: only patients with both timepoints enter a
    pairwise contrast, patient is a fixed-effect covariate and timepoint labels are permuted
    within patient. Effect sizes are paired (mean difference, matched-pairs rank-biserial).
  * Multiple testing: Benjamini-Hochberg / Holm / Bonferroni per family (see finalize()).
"""
from __future__ import annotations

import glob
import itertools
import os
import re
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

# pandas 2.x: opt in to the pandas 3 behaviour (no silent downcasting in replace / where / fillna), so the app
# behaves the same on both versions and shows no FutureWarnings. pandas 3 has this behaviour built in.
if int(pd.__version__.split(".")[0]) < 3:
    try:
        pd.set_option("future.no_silent_downcasting", True)
    except Exception:                                   # very old pandas without the option
        pass

SITES = ["non-lesional", "lesional", "healthy", "nostril", "gut"]
SKIN_TYPES = ["moist", "sebaceous", "dry"]
HEALTHY = "healthy"
EXCLUDE_TAXA = ("Unclassified", "Other")
SITE_ALIASES = {"nonlesional": "non-lesional", "non-lesion": "non-lesional",
                "nl": "non-lesional", "l": "lesional", "lesion": "lesional",
                "nose": "nostril", "nasal": "nostril", "nares": "nostril",
                # 1806: TL/NL = target lesion / non-lesional (psoriasis patients)
                "tl": "lesional",
                "hvnl": "non-lesional", "hv_nl": "non-lesional", "hv-nl": "non-lesional",
                "hvtl": "lesional", "hv_tl": "lesional", "hv-tl": "lesional",
                "hs": "lesional",
                "x1": "lesional", "x2": "lesional", "le2": "lesional",
                # other common spellings (e.g. the Site column of the blister-fluid file)
                "ls": "lesional", "les": "lesional", "lesional-skin": "lesional", "involved": "lesional",
                "pp": "lesional", "laesionaal": "lesional",
                "perilesional": "lesional", "peri-lesional": "lesional", "peri": "lesional",
                "blister-perilesional": "lesional", "blister-non-lesional": "non-lesional",
                "nls": "non-lesional", "non-les": "non-lesional", "nonles": "non-lesional",
                "non-lesional-skin": "non-lesional", "uninvolved": "non-lesional", "pn": "non-lesional",
                "non-laesionaal": "non-lesional", "niet-laesionaal": "non-lesional"}
# NOTE: the skin samples of healthy volunteers get the site 'healthy'. Where the label says which patient site
# it matches (HV_TL / HS = target, lesional-matched; HV_NL = non-lesional) that is kept in 'healthy_match':
# such a sample is the healthy reference for that site only; a sample without a match (e.g. a healthy
# volunteer's blister) is the reference for both the lesional and the non-lesional site.
# X1 = L1, X2/LE2 = L2 (second, less-severe lesion); lesion_rank orders them.
HEALTHY_LABELS = {"healthy", "healthy-control", "control", "hc", "healthy-skin", "healthy control"}
SKIN_ALIASES = {"moist": "moist", "wet": "moist", "sebaceous": "sebaceous", "seb": "sebaceous",
                "oily": "sebaceous", "dry": "dry"}
TIMEPOINT_FIRST = ["screening", "baseline", "bl", "t0", "pre", "before", "pretreatment", "pre-treatment"]


# =============================================================== loading
def load_emu_rank(results_dir, rank):
    """Samples x taxa relative abundance from Emu *_rel-abundance.tsv files at one rank."""
    files = sorted(glob.glob(os.path.join(results_dir, "**", "*_rel-abundance.tsv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No *_rel-abundance.tsv files under {results_dir}")
    cols = {}
    for fp in files:
        sample = os.path.basename(fp).replace("_rel-abundance.tsv", "")
        df = pd.read_csv(fp, sep="\t")
        if rank not in df.columns:
            raise KeyError(f"Column '{rank}' not in {fp}. Available: {list(df.columns)}")
        df = df[df["tax_id"].astype(str) != "unassigned"].copy()
        df[rank] = df[rank].fillna("Unclassified")
        cols[sample] = df.groupby(rank)["abundance"].sum()
    mat = pd.DataFrame(cols).fillna(0).T
    return mat.div(mat.sum(axis=1), axis=0)


def standardize_metadata(df, study, column_map=None, pasi_cutoffs=(10, 20)):
    """Map one study's metadata onto the standard columns. Returns (md, notes)."""
    notes = []
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if column_map:
        df = df.rename(columns={k.lower(): v for k, v in column_map.items()})
    for req in ("sample", "site"):
        if req not in df.columns:
            raise ValueError(f"[{study}] metadata has no '{req}' column (use column_map in the config)")
    for c in ("patient", "group", "severity", "skin_type", "pasi", "body_site", "timepoint", "lesion_rank",
              "local_severity", "global_severity"):
        if c not in df.columns:
            df[c] = np.nan
    if "lesion" not in df.columns:
        df["lesion"] = ""
    df["lesion"] = df["lesion"].astype(object).where(df["lesion"].notna(), "")
    df["sample"] = df["sample"].astype(str).str.strip()
    timepoint = df["timepoint"].astype("string").str.strip().str.lower()
    df["timepoint"] = timepoint.replace("", pd.NA)

    raw = df["site"].astype(str).str.strip().str.lower().str.replace(r"[ _]", "-", regex=True)
    healthy_site = raw.isin(HEALTHY_LABELS)
    site = raw.replace(SITE_ALIASES)
    site[healthy_site] = "healthy"
    bad = sorted(set(site) - set(SITES))
    if bad:
        raise ValueError(f"[{study}] unrecognised site label(s) {bad}; expected {SITES} (or 'healthy')")
    df["site"] = site

    group = df["group"].astype(object).where(df["group"].notna(), "unspecified").astype(str).str.strip()
    group = group.where(~group.str.lower().isin(HEALTHY_LABELS), HEALTHY)
    group = group.where(~healthy_site, HEALTHY)
    df["group"] = group
    df["site"], df["healthy_match"] = to_healthy_site(df["site"], group)
    not_lesional = df["site"].ne("lesional")              # a lesion score belongs to a lesional sample only
    df["lesion_rank"] = pd.to_numeric(df["lesion_rank"], errors="coerce").mask(not_lesional)
    df["local_severity"] = df["local_severity"].mask(not_lesional)

    sev = blank_to_nan(df["severity"].astype(object).where(df["severity"].notna(), "").astype(str).str.lower())
    pasi = pd.to_numeric(df["pasi"], errors="coerce")
    need = sev.isna() & pasi.notna()
    if need.any():
        c1, c2 = pasi_cutoffs
        binned = pd.cut(pasi, [-np.inf, c1, c2, np.inf], labels=["mild", "moderate", "severe"],
                        right=False).astype(object)
        sev[need] = binned[need]
        notes.append(f"[{study}] severity derived from PASI (mild <{c1}, moderate {c1}-<{c2}, severe >={c2}) "
                     f"for {int(need.sum())} samples")
    sev = sev.where(df["group"] != HEALTHY, HEALTHY)
    if sev.isna().any():
        notes.append(f"[{study}] {int(sev.isna().sum())} samples have no severity -> 'unspecified'")
        sev = sev.fillna("unspecified")
    df["severity"] = sev
    df["pasi"] = pasi
    for c in ("local_severity", "global_severity"):         # numeric scores; blank = no value
        df[c] = pd.to_numeric(df[c], errors="coerce")

    st = df["skin_type"].astype(object).where(df["skin_type"].notna(), "nan").astype(str)
    st = st.str.strip().str.lower().map(SKIN_ALIASES)
    unk = df["skin_type"].notna() & st.isna()
    if unk.any():
        notes.append(f"[{study}] {int(unk.sum())} samples with unrecognised skin_type set to missing")
    df["skin_type"] = st

    df["patient"] = df["patient"].astype(object).where(df["patient"].notna(), df["sample"]).astype(str)
    bad = df.groupby("patient")["severity"].nunique()
    if (bad > 1).any():
        # A patient's stage may differ over time (or across lesions) and the per-sample value is the
        # authority, so keep each sample's own severity; only list the patients whose stages differ.
        n = int((bad > 1).sum())
        notes.append(f"[{study}] {n} patient(s) have more than one severity across their samples (e.g. "
                     f"{list(bad[bad > 1].index)[:5]}); each sample keeps its own stage")
    df["study"] = study
    df["patient"] = study + "|" + df["patient"]
    df["body_site"] = df["body_site"].astype(object).where(df["body_site"].notna(), "")
    df["uid"] = study + "|" + df["sample"]
    return df.set_index("uid")[["study", "sample", "patient", "site", "group", "severity",
                                "skin_type", "timepoint", "pasi", "body_site", "lesion_rank", "lesion",
                                "local_severity", "global_severity", "healthy_match"]], notes


def load_studies(config, rank, base_dir="."):
    """config = {'studies': {name: {results_dir, metadata, column_map?}}, 'pasi_cutoffs'?}.
    Relative paths are resolved against base_dir (the folder of the config file).
    Returns (mat, md, notes)."""
    def path(p):
        return p if os.path.isabs(p) else os.path.join(base_dir, p)

    mats, mds, notes = [], [], []
    for name, spec in config["studies"].items():
        mdf = pd.read_csv(path(spec["metadata"]), sep=None, engine="python", dtype=str)
        md, n = standardize_metadata(mdf, str(name), spec.get("column_map"),
                                     tuple(config.get("pasi_cutoffs", (10, 20))))
        notes += n
        mat = load_emu_rank(path(spec["results_dir"]), rank)
        have = md["sample"].isin(mat.index)
        if (~have).any():
            notes.append(f"[{name}] {int((~have).sum())} metadata samples without Emu output were dropped")
        for s in sorted(set(mat.index) - set(md["sample"])):
            notes.append(f"[{name}] Emu sample '{s}' has no metadata and was dropped")
        md = md[have]
        mat = mat.loc[md["sample"].values]
        mat.index = md.index
        mats.append(mat)
        mds.append(md)
    mat = pd.concat(mats).fillna(0)
    mat = mat.div(mat.sum(axis=1), axis=0)
    md = pd.concat(mds)
    return mat, md, notes


# =============================================================== level orders / palette
def _natural_key(text):
    # tag digits vs text so the key is always comparable (avoids int/str TypeError on labels like "-14")
    return [(0, int(t)) if t.isdigit() else (1, t) for t in re.split(r"(\d+)", str(text))]


TIMEPOINT_ORDER = ["screening", "baseline", "mid", "follow-up"]   # canonical visit labels, earliest first


def timepoint_levels(md, order=None):
    """Ordered timepoint labels: configured order first, then baseline-like names, then natural
    sort (so 'week 4' < 'week 12' and 'pre' always precedes 'post' -> contrast 'post vs pre')."""
    present = list(pd.unique(md["timepoint"].dropna()))
    order = [str(o).strip().lower() for o in (order or TIMEPOINT_ORDER)]
    out = [t for t in order if t in present]
    rest = [t for t in present if t not in out]
    first = [t for t in TIMEPOINT_FIRST if t in rest]
    rest = first + sorted((t for t in rest if t not in first), key=_natural_key)
    return out + rest


SEVERITY_ORDER: list[str] = []      # optional global order, mildest -> most severe (lower-case labels)


# Known ordinal scales, mildest first. Each scale runs blue -> red on its own, so 'mild' psoriasis and stage IA
# are both blue and 'severe' and the highest stage present are both red. Set SEVERITY_ORDER (above) to use one
# single order across all labels instead.
SEVERITY_SCALES = (("ia", "ib", "iia", "iib", "iii", "iiia", "iiib", "iv", "iva", "iva1", "iva2", "ivb", "sezary", "ss"),
                   ("mild", "moderate", "severe"))


def severity_scale(level):
    """Index of the ordinal scale a label belongs to (None if unknown)."""
    for i, scale in enumerate(SEVERITY_SCALES):
        if level in scale:
            return i
    return None


def severity_levels(md, order=None):
    """'healthy' first, then SEVERITY_ORDER (or `order`), then the known scales, then other labels A-Z."""
    order = SEVERITY_ORDER if order is None else order
    present = list(pd.unique(md["severity"]))
    present = [s for s in present if s is not None and not (isinstance(s, float) and np.isnan(s))]
    out = [HEALTHY] if HEALTHY in present else []
    out += [s for s in (order or []) if s in present and s not in out]
    out += [s for scale in SEVERITY_SCALES for s in scale if s in present and s not in out]
    out += sorted(s for s in present if s not in out and isinstance(s, str))
    return out


def severity_colors(levels):
    """healthy = green; blue (mildest) -> red (most severe) within each scale; 'unspecified' = grey."""
    lo, hi = np.array([43, 108, 176]), np.array([192, 57, 43])
    cols = {HEALTHY: "#2E9E4F"}
    sick = [s for s in levels if s not in (HEALTHY, "unspecified")]
    if SEVERITY_ORDER:
        groups = [sick]
    else:
        groups = [[s for s in scale if s in sick] for scale in SEVERITY_SCALES]
        groups.append([s for s in sick if severity_scale(s) is None])
    for group in groups:
        for i, s in enumerate(group):
            t = i / (len(group) - 1) if len(group) > 1 else 0.0
            r, g, b = (lo + t * (hi - lo)).round().astype(int)
            cols[s] = f"#{r:02x}{g:02x}{b:02x}"
    if "unspecified" in levels:
        cols["unspecified"] = "#9a9a9a"
    return cols


# =============================================================== transforms
def clr(mat, pseudo=None):
    x = mat.values.astype(float)
    if pseudo is None:
        pseudo = x[x > 0].min() / 2
    x = x + pseudo
    x = x / x.sum(axis=1, keepdims=True)
    lx = np.log(x)
    return pd.DataFrame(lx - lx.mean(axis=1, keepdims=True), index=mat.index, columns=mat.columns)


def clr_by_compartment(mat, md, pseudo=None):
    """CLR computed separately for gut samples and for skin/nostril samples, each over the taxa
    detected in that compartment (so thousands of taxa that cannot occur there do not distort the
    centring). Call it ONCE on the full matrix, then subset rows, so values never depend on the
    current filters. Taxa absent from a compartment get 0 (they are never tested there)."""
    if pseudo is None:
        pseudo = float(mat.values[mat.values > 0].min() / 2)
    out = pd.DataFrame(0.0, index=mat.index, columns=mat.columns)
    is_gut = md.reindex(mat.index)["site"].eq("gut").values
    for flag in (False, True):
        rows = mat.index[is_gut == flag]
        if len(rows) == 0:
            continue
        sub = mat.loc[rows]
        cols = sub.columns[(sub > 0).any(axis=0)]
        out.loc[rows, cols] = clr(sub[cols], pseudo).values
    return out


def shannon(mat):
    p = mat.values
    return pd.Series(-np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0), axis=1),
                     index=mat.index)


def pcoa(D):
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    w, v = np.linalg.eigh(-0.5 * J @ (D ** 2) @ J)
    idx = np.argsort(w)[::-1]
    w, v = w[idx], v[:, idx]
    pos = w > 1e-10
    return v[:, pos] * np.sqrt(w[pos]), w[pos] / w[pos].sum()


def distance_matrices(mat, clr_mat):
    return {"braycurtis": squareform(pdist(mat.values, "braycurtis")),
            "aitchison": squareform(pdist(clr_mat.values, "euclidean"))}


# =============================================================== multiple testing
def adjust_p(p, method="bh"):
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    pv, m = p[ok], int(ok.sum())
    if m == 0:
        return out
    if method == "bonferroni":
        q = np.minimum(pv * m, 1.0)
    elif method == "holm":
        order = np.argsort(pv)
        adj = np.maximum.accumulate((m - np.arange(m)) * pv[order])
        q = np.empty(m)
        q[order] = np.minimum(adj, 1.0)
    elif method == "bh":
        order = np.argsort(pv)[::-1]
        adj = np.minimum.accumulate(pv[order] * m / np.arange(m, 0, -1))
        q = np.empty(m)
        q[order] = np.minimum(adj, 1.0)
    else:
        raise ValueError(method)
    out[ok] = q
    return out


def add_q(df, group_cols, method, col="q"):
    df = df.reset_index(drop=True).copy()
    df[col] = np.nan
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        df[col] = adjust_p(df["p"].values, method)
        return df
    for _, g in df.groupby(group_cols, dropna=False):
        df.loc[g.index, col] = adjust_p(g["p"].values, method)
    return df


# =============================================================== permutation core
def gower(D):
    A = -0.5 * D ** 2
    return A - A.mean(0, keepdims=True) - A.mean(1, keepdims=True) + A.mean()


def _dummies(codes, k):
    return np.eye(k)[np.asarray(codes).astype(int)][:, 1:]


def cov_dummies(meta, names):
    """Dummy-coded covariate matrix (no intercept); NaN treated as its own level."""
    cols = []
    for c in names:
        cat = pd.Categorical(meta[c].fillna("n/a"))
        if len(cat.categories) > 1:
            cols.append(_dummies(cat.codes, len(cat.categories)))
    return np.hstack(cols) if cols else np.zeros((len(meta), 0))


def _basis(X):
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    r = int((s > s.max() * 1e-10).sum())
    return U[:, :r], r


def make_permuter(codes, scheme, strata, blocks, rng):
    """Function returning a permuted copy of label vector `codes`.
    free    : rows exchangeable (within `blocks` if given)
    within  : permute labels inside each stratum (patient)  -> within-patient factors
    between : permute whole patients (label constant within patient), within `blocks`"""
    blocks = None if blocks is None else np.asarray(blocks, dtype=object)
    strata = None if strata is None else np.asarray(strata, dtype=object)
    if scheme == "free":
        if blocks is None:
            return lambda: rng.permutation(codes)
        bidx = [np.where(blocks == b)[0] for b in pd.unique(blocks)]

        def free_blocked():
            out = codes.copy()
            for ix in bidx:
                out[ix] = rng.permutation(codes[ix])
            return out
        return free_blocked
    pat_idx, uniq = pd.factorize(pd.Series(strata))
    if scheme == "within":
        groups = [np.where(pat_idx == i)[0] for i in range(len(uniq))]

        def within():
            out = codes.copy()
            for g in groups:
                out[g] = rng.permutation(codes[g])
            return out
        return within
    if scheme == "between":
        s = pd.Series(codes)
        if s.groupby(pat_idx).nunique().max() > 1:
            raise ValueError("Factor is not constant within patient: cannot permute whole patients.")
        first = s.groupby(pat_idx).first().values
        if blocks is None:
            P = len(first)
            return lambda: first[rng.permutation(P)][pat_idx]
        pb = pd.Series(blocks).groupby(pat_idx).first().values
        bidx = [np.where(pb == b)[0] for b in pd.unique(pb)]

        def between_blocked():
            nf = first.copy()
            for ix in bidx:
                nf[ix] = first[rng.permutation(ix)]
            return nf[pat_idx]
        return between_blocked
    raise ValueError(scheme)


def perm_test(data, codes, k, cov, scheme="free", strata=None, blocks=None, n_perm=999,
              seed=0, kind="feat"):
    """Marginal permutation F test for one term, adjusted for covariates `cov`.
    kind='dist': data = distance matrix (PERMANOVA)  -> scalar results
    kind='feat': data = n x m matrix (already rank-transformed) -> arrays of length m
    k = number of levels of a categorical term, or None for a numeric (1 df) term."""
    n = len(codes)
    if kind == "dist":
        G = gower(data)
        sst = np.array([np.trace(G)])

        def ss(U):
            return np.array([np.sum(U * (G @ U))])
    else:
        Yc = data - data.mean(0, keepdims=True)
        sst = (Yc ** 2).sum(0)

        def ss(U):
            return ((U.T @ Yc) ** 2).sum(0)
    X0 = np.hstack([np.ones((n, 1)), cov])
    U0, r0 = _basis(X0)
    ss0 = ss(U0)

    def stat(cd):
        tc = cd.reshape(-1, 1).astype(float) if k is None else _dummies(cd, k)
        Uf, rf = _basis(np.hstack([X0, tc]))
        dft, dfr = rf - r0, n - rf
        ssf = ss(Uf)
        num, den = ssf - ss0, sst - ssf
        with np.errstate(divide="ignore", invalid="ignore"):
            F = np.where((dft > 0) & (dfr > 0) & (den > 1e-12 * np.maximum(sst, 1e-300)),
                         (num / max(dft, 1)) / (den / max(dfr, 1)), np.nan)
        return F, num, dft, dfr

    F, num, dft, dfr = stat(codes)
    with np.errstate(divide="ignore", invalid="ignore"):
        R2 = num / sst
    permute = make_permuter(codes, scheme, strata, blocks, np.random.default_rng(seed))
    Fp = np.empty((n_perm, F.size))
    for i in range(n_perm):
        Fp[i] = stat(permute())[0]
    with np.errstate(invalid="ignore"):
        cnt = np.sum(Fp >= F * (1 - 1e-9), axis=0)
    p = (cnt + 1) / (n_perm + 1)
    p[np.isnan(F)] = np.nan
    if kind == "dist":
        return dict(F=F[0], R2=R2[0], df=dft, df_res=dfr, p=p[0])
    return dict(F=F, R2=R2, df=dft, df_res=dfr, p=p)


def permdisp(D, codes, k, strata, blocks, n_perm, seed):
    """PERMDISP: ANOVA F on distances to group centroids; whole patients permuted."""
    n = len(codes)
    nan = dict(F=np.nan, R2=np.nan, df=k - 1, df_res=n - k, p=np.nan)
    if k < 2 or n <= k:
        return nan
    coords, _ = pcoa(D)

    def fstat(cd):
        z = np.empty(n)
        for g in range(k):
            m = cd == g
            if m.any():
                z[m] = np.linalg.norm(coords[m] - coords[m].mean(0), axis=1)
        gm = z.mean()
        ssb = sum((cd == g).sum() * (z[cd == g].mean() - gm) ** 2 for g in range(k) if (cd == g).any())
        ssw = sum(((z[cd == g] - z[cd == g].mean()) ** 2).sum() for g in range(k) if (cd == g).any())
        return (ssb / (k - 1)) / (ssw / (n - k)) if ssw > 0 else np.nan

    F = fstat(codes)
    if np.isnan(F):
        return nan
    permute = make_permuter(codes, "between", strata, blocks, np.random.default_rng(seed))
    Fp = np.array([fstat(permute()) for _ in range(n_perm)])
    p = (np.nansum(Fp >= F * (1 - 1e-9)) + 1) / (n_perm + 1)
    return dict(F=F, R2=np.nan, df=k - 1, df_res=n - k, p=p)


# =============================================================== settings / selection
@dataclass(frozen=True)
class Settings:
    studies: tuple = ()
    sites: tuple = tuple(SITES)
    groups: tuple = ()
    severities: tuple = ()
    timepoints: tuple = ()            # empty = no timepoint filter (samples without timepoint always kept)
    skin_mode: str = "off"            # off | restrict | stratify | adjust
    skin_types: tuple = tuple(SKIN_TYPES)
    study_mode: str = "separate"      # separate | adjust
    group_mode: str = "separate"      # separate | adjust
    correction: str = "bh"
    q_thr: float = 0.05
    n_perm: int = 999
    seed: int = 42
    min_prev: float = 0.2
    min_abund: float = 0.001
    permdisp: bool = True
    lesions: str = "most severe"      # most severe | both  (statistics always use the most severe lesion)
    timepoint_mode: str = "separate"  # separate: one block per timepoint | average: patient mean over timepoints

    def key(self):
        return str(sorted(asdict(self).items()))


def most_severe_lesion(md):
    """Keep, per patient, site and timepoint, only the most severe lesion (lesion_rank 1). Rows without a rank
    (non-lesional, nostril, gut, other modalities) are kept."""
    if "lesion_rank" not in md or md["lesion_rank"].isna().all():
        return md
    rank = md["lesion_rank"].astype(float)
    tp = md["timepoint"].astype(object).where(md["timepoint"].notna(), "")
    best = rank.groupby([md["patient"].values, md["site"].values, tp.values]).transform("min")
    return md[rank.isna() | (rank <= best)]


def local_severity_grouping(values, max_groups=8, n_bins=4):
    """Groups for plotting by local severity. Up to max_groups distinct values: one group per value; more: n_bins
    equal-width ranges. Returns (function mapping a Series of values to group labels, ordered list of labels)."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if v.empty:
        return (lambda s: pd.Series(np.nan, index=getattr(s, "index", None), dtype=object)), []
    fmt = lambda x: f"{x:g}"
    distinct = np.sort(v.unique())
    if len(distinct) <= max_groups:
        levels = [fmt(x) for x in distinct]
        return (lambda s: pd.to_numeric(s, errors="coerce").map(lambda x: np.nan if pd.isna(x) else fmt(x))), levels
    edges = np.linspace(distinct[0], distinct[-1], n_bins + 1)
    levels = [f"{fmt(round(a, 2))}–{fmt(round(b, 2))}" for a, b in zip(edges[:-1], edges[1:])]

    def to_group(s):
        x = pd.to_numeric(s, errors="coerce")
        idx = np.clip(np.searchsorted(edges, x.values, side="right") - 1, 0, n_bins - 1)
        return pd.Series([np.nan if pd.isna(xi) else levels[i] for xi, i in zip(x.values, idx)], index=x.index,
                         dtype=object)
    return to_group, levels


def multiple_lesions(md):
    """True if some patient has a second, less severe lesion at a timepoint."""
    if "lesion_rank" not in md or md["lesion_rank"].isna().all():
        return False
    return bool((md["lesion_rank"] > 1).any())             # a less severe lesion exists somewhere


def timeless_healthy(md):
    """Mask of the samples of healthy volunteers who were sampled at ONE timepoint only. Their sample is not a
    'baseline': it has no place on the study timeline and serves as the healthy reference at every timepoint.
    A healthy volunteer with several timepoints (baseline and follow-up) keeps them like a patient.
    The loaded datasets carry the answer in the column 'healthy_reference', decided on ALL their samples, so a
    volunteer with two timepoints does not become a reference when one of them is filtered out."""
    if "healthy_reference" in md:
        return md["healthy_reference"].astype(bool).rename(None)
    if "timepoint" not in md or md.empty:
        return pd.Series(False, index=md.index)
    n_tp = md["timepoint"].astype(object).groupby(md["patient"].astype(str).values).nunique()
    single = md["patient"].astype(str).map(n_tp).fillna(0).le(1)
    return (md["group"].eq(HEALTHY) & single.values).rename(None)


def filter_md(md, S):
    m = md[md["study"].isin(S.studies) & md["site"].isin(S.sites)
           & md["group"].isin(S.groups) & md["severity"].isin(S.severities)]
    if "timepoint" in m and S.timepoints:                  # keep studies w/o timepoints and the healthy reference
        m = m[m["timepoint"].isna() | m["timepoint"].isin(S.timepoints) | timeless_healthy(m)]
    if S.skin_mode != "off":
        keep = m["skin_type"].isin(S.skin_types) | (~m["site"].isin(["lesional", "non-lesional"]) & m["skin_type"].isna())
        m = m[keep]
    if S.lesions == "most severe":
        m = most_severe_lesion(m)
    return m


@dataclass
class Block:
    label: str
    names: list
    cov_candidates: list
    perm_block: str | None


def make_blocks(md, S, factor="severity", min_samples=4):
    """Split the selected samples into analysis blocks (see module docstring)."""
    md = most_severe_lesion(filter_md(md, S))     # statistics: one lesion per patient, the most severe
    if factor == "timepoint":                      # studies / samples without timepoints cannot be compared
        md = md[md["timepoint"].notna()]
    cov, perm_block = [], None

    def split(frames, col, values):
        out = []
        for lab, f in frames:
            for v in values:
                sub = f[f[col].isna()] if v is None else f[f[col] == v]
                if len(sub):
                    name = "non-skin sites (nostril, gut)" if v is None else str(v)
                    out.append((f"{lab} | {name}" if lab else name, sub))
        return out

    if factor in ("severity", "timepoint", "site", "local_severity"):
        disease = [g for g in pd.unique(md["group"]) if g != HEALTHY]
        if S.group_mode == "separate" and disease:
            frames = []
            for g in disease:
                gm = md[md["group"] == g]
                if factor == "severity":           # healthy controls of the same studies serve as the reference level
                    hm = md[(md["group"] == HEALTHY) & md["study"].isin(gm["study"].unique())]
                    gm = pd.concat([gm, hm])
                frames.append((str(g), gm))
        elif S.group_mode == "adjust" and disease:
            frames = [("", md[md["group"] != HEALTHY])]
            cov.append("group")
        else:
            frames = [("", md)]
    else:
        frames = [("", md)]

    if factor != "study":
        if S.study_mode == "separate":
            frames = split(frames, "study", sorted(md["study"].unique()))
        else:
            cov.append("study")
            perm_block = "study"
    if factor != "timepoint" and S.timepoint_mode == "separate" and md["timepoint"].nunique() > 1:
        ref = timeless_healthy(md)                         # healthy sampled once: reference at every timepoint
        tps = timepoint_levels(md[~ref]) or timepoint_levels(md)
        frames = [(f"{lab} | {t}" if lab else str(t), f[(f["timepoint"] == t) | ref.reindex(f.index)])
                  for lab, f in frames for t in tps      # never mix the timepoints of patients
                  if ((f["timepoint"] == t) & ~ref.reindex(f.index)).any() or ref.reindex(f.index).all()]
    if S.skin_mode == "stratify":
        frames = split(frames, "skin_type", [t for t in SKIN_TYPES if t in S.skin_types] + [None])
    elif S.skin_mode == "adjust":
        cov.append("skin_type")

    blocks = []
    for lab, f in frames:
        if len(f) >= min_samples:
            blocks.append(Block(lab or "all selected samples", list(f.index), list(cov), perm_block))
    return blocks


# =============================================================== units / strata
# Skin (or blister) samples of healthy volunteers have the site 'healthy'. They are the healthy reference in
# the lesional and the non-lesional stratum (and in pooled), and form no stratum of their own when the patients
# have lesional / non-lesional samples. md['healthy_match'] limits a sample to one of the two strata.
SHARED_HEALTHY_SITES = {"healthy"}


def healthy_reference(md, stratum):
    """Mask of the healthy-site samples that are the reference for `stratum` ('lesional' | 'non-lesional' |
    'pooled'): all of them, except a sample matched to the other site."""
    ref = md["site"].isin(SHARED_HEALTHY_SITES)
    if stratum in ("lesional", "non-lesional") and "healthy_match" in md:
        other = "non-lesional" if stratum == "lesional" else "lesional"
        ref = ref & md["healthy_match"].astype(object).fillna("").ne(other)
    return ref


def blank_to_nan(s: pd.Series, extra=()) -> pd.Series:
    """Text column with blank / 'nan' / 'none' (and `extra`) cells set to missing. Always returns object dtype:
    pandas never silently turns the column into numbers (no FutureWarning on pandas 2.x, same result on 3.x)."""
    t = s.astype(object)
    txt = t.where(t.notna(), "").astype(str).str.strip()
    missing = (txt.eq("") | txt.str.lower().isin({"nan", "none", "<na>", *extra})).to_numpy()
    arr = txt.to_numpy(dtype=object, copy=True)
    arr[missing] = np.nan
    return pd.Series(arr, index=s.index, name=s.name, dtype=object)


def to_healthy_site(site, group, sites=("lesional", "non-lesional")):
    """Skin samples of healthy volunteers -> site 'healthy'. Returns (site, healthy_match): the patient site a
    healthy sample was matched to ('lesional' for HV_TL / HS, 'non-lesional' for HV_NL), '' if none."""
    site = pd.Series(site, copy=True).astype(object)
    healthy = pd.Series(group, index=site.index).astype(object).eq(HEALTHY)
    move = healthy & site.isin(sites)
    match = pd.Series("", index=site.index, dtype=object)
    match[move] = site[move].where(site[move].isin(["lesional", "non-lesional"]), "")
    site[move] = "healthy"
    return site, match


def strata_for(sites):
    s = [x for x in ("non-lesional", "lesional") if x in sites]
    if not s and "healthy" in sites:                       # no patient skin sites: healthy is its own stratum
        s.append("healthy")
    if len(s) == 2:
        s.append("pooled")
    if "nostril" in sites:
        s.append("nostril")
    if "gut" in sites:
        s.append("gut")
    s += sorted(x for x in sites if x not in SITES)      # other specimen types (plasma, blister, ...)
    return s


STRATUM_SITES = {"non-lesional": ["non-lesional"], "lesional": ["lesional"],
                 "pooled": ["lesional", "non-lesional"], "nostril": ["nostril"], "gut": ["gut"]}


def stratum_samples(md, stratum, factor=None, shared=True):
    """Samples of a stratum. 'pooled' keeps patients with BOTH lesional and non-lesional samples
    (at the same timepoint when factor == 'timepoint'). shared: the lesional and the non-lesional stratum
    include the healthy volunteers' samples at a SHARED_HEALTHY_SITES site (their reference for both)."""
    in_site = md["site"].isin(STRATUM_SITES.get(stratum, [stratum]))
    ref = healthy_reference(md, stratum) if shared and stratum in ("lesional", "non-lesional", "pooled") else None
    if ref is not None and stratum != "pooled":
        in_site = in_site | ref
    sel = md[in_site]
    if factor == "timepoint":
        sel = sel[sel["timepoint"].notna()]
    if stratum == "pooled":
        unit = ["patient", "timepoint"] if factor == "timepoint" else ["patient"]
        n_sites = sel.groupby(unit)["site"].nunique()
        complete = n_sites[n_sites == 2].index
        if factor == "timepoint":
            keys = pd.MultiIndex.from_frame(sel[unit])
            sel = sel[keys.isin(complete)]
        else:
            sel = sel[sel["patient"].isin(complete)]
        if ref is not None:                                # healthy reference joins the pooled stratum
            add = md[ref]
            if factor == "timepoint":
                add = add[add["timepoint"].notna()]
            sel = pd.concat([sel, add])
    return sel.index


def unit_frame(md, values, stratum, factor=None, keep_lesions=False, shared=True):
    """One row per patient, or per patient-timepoint when factor == 'timepoint'
    (mean over the unit's samples in the stratum). Y and meta are built from the same
    sorted keys, so rows are aligned. keep_lesions (plots only): a patient's lesions stay separate rows.
    Returns (values DataFrame, meta DataFrame)."""
    names = stratum_samples(md, stratum, factor, shared=shared)
    if len(names) == 0:
        return None, None
    m = md.loc[names].sort_values("site")          # lesional first -> pooled takes its skin type
    v = values.loc[m.index]
    group_cols = ["patient", "timepoint"] if factor == "timepoint" else ["patient"]
    if keep_lesions and stratum == "lesional" and "lesion" in m:
        group_cols.append("lesion")
    key_cols = [f"__{c}" for c in group_cols]
    Y = v.copy()
    for c, kc in zip(group_cols, key_cols):
        Y[kc] = m[c].astype(object).values
    Y = Y.groupby(key_cols, sort=True).mean()
    extra = {c: (c, how) for c, how in (("local_severity", "max"), ("global_severity", "mean")) if c in m}
    meta = m.assign(**{c: m[c].astype(object) for c in group_cols}).groupby(group_cols, sort=True).agg(
        severity=("severity", "first"), study=("study", "first"), group=("group", "first"),
        skin_type=("skin_type", "first"), body_site=("body_site", lambda s: " / ".join(map(str, s))),
        samples=("sample", lambda s: ", ".join(s)), **extra)
    meta = meta.reset_index()
    meta["site"] = stratum
    if stratum in ("lesional", "non-lesional", "pooled"):      # healthy reference keeps its own marker
        healthy_ref = set(m.loc[m["site"].isin(SHARED_HEALTHY_SITES), "patient"])
        meta.loc[meta["patient"].isin(healthy_ref).values, "site"] = "healthy"
    Y.index = pd.RangeIndex(len(Y))
    meta.index = Y.index
    return Y, meta


# =============================================================== paired helpers
def _patients_with_levels(meta, factor, lv, need):
    """Boolean mask of units whose patient has >= `need` distinct levels among `lv`."""
    inlv = meta[factor].isin(lv)
    cnt = meta[inlv].groupby("patient")[factor].nunique()
    return (inlv & meta["patient"].isin(cnt[cnt >= need].index)).values


def _paired_effect(d):
    """Matched-pairs rank-biserial correlation for each column of d (pairs x features)."""
    out = np.zeros(d.shape[1])
    for j in range(d.shape[1]):
        x = d[:, j]
        x = x[x != 0]
        if len(x):
            r = rankdata(np.abs(x))
            wp, wn = r[x > 0].sum(), r[x < 0].sum()
            out[j] = (wp - wn) / (wp + wn)
    return out


# =============================================================== univariate tests
def feature_tests(Y, meta, factor, levels, ordered, cov_candidates, perm_block, S, seeds,
                  rel=None, min_n=2, min_pairs=3):
    """Omnibus (>2 levels), ordinal trend (ordered, >2 levels) and all pairwise contrasts
    for every column of Y. Rank-based permutation tests adjusted for covariates.
    factor == 'timepoint' -> paired design (patient covariate, within-patient permutations,
    only patients observed at both timepoints, paired effect sizes)."""
    Y = Y.reset_index(drop=True)
    meta = meta.reset_index(drop=True)
    paired = factor == "timepoint"
    lab = meta[factor].values
    present = [l for l in levels if (lab == l).sum() >= min_n]
    if len(present) < 2:
        return []
    jobs = []
    if len(present) > 2:
        jobs.append(("omnibus", "all", None, None, present))
        if ordered:
            jobs.append(("trend", "trend", None, None, present))
    for lo, hi in itertools.combinations(present, 2):
        jobs.append(("pairwise", f"{hi} vs {lo}", lo, hi, [lo, hi]))
    rows = []
    feats = list(Y.columns)
    for test, contrast, lo, hi, lv in jobs:
        if paired:      # only patients observed at both (pairwise) / >= 2 (omnibus, trend) timepoints
            mask = _patients_with_levels(meta, factor, lv, len(lv) if test == "pairwise" else 2)
        else:
            mask = np.isin(lab, lv)
        pos = np.flatnonzero(mask)
        sub = meta.iloc[pos]
        if paired and sub["patient"].nunique() < min_pairs:
            continue
        Ys = Y.values[pos]
        R = rankdata(Ys, axis=0)
        codes = pd.Categorical(sub[factor], categories=lv).codes.astype(int)
        covs = [c for c in cov_candidates if c in sub.columns and sub[c].fillna("n/a").nunique() > 1]
        if paired:
            covs = ["patient"] + [c for c in covs if c != "patient"]
        cov = cov_dummies(sub, covs)
        blocks = sub[perm_block].values if (perm_block in covs) else None
        scheme = "within" if paired else "free"
        perm_strata = sub["patient"].values if paired else None
        if test == "trend":
            r = perm_test(R, codes.astype(float), None, cov, scheme, perm_strata, blocks, S.n_perm,
                          next(seeds), "feat")
            rc = rankdata(codes)
            Rc = R - R.mean(0)
            with np.errstate(divide="ignore", invalid="ignore"):
                eff = (Rc.T @ (rc - rc.mean())) / (np.sqrt((Rc ** 2).sum(0)) * np.sqrt(((rc - rc.mean()) ** 2).sum()))
        else:
            r = perm_test(R, codes, len(lv), cov, scheme, perm_strata, blocks, S.n_perm,
                          next(seeds), "feat")
            eff = r["R2"]
        if test == "pairwise":
            if paired:                                           # paired effect sizes (hi - lo per patient)
                ih = sub[sub[factor] == hi].sort_values("patient").index.values
                il = sub[sub[factor] == lo].sort_values("patient").index.values
                d = Y.values[ih] - Y.values[il]
                nh = nl = len(ih)
                diff, eff = d.mean(0), _paired_effect(d)
                med_h = np.median(rel.values[ih], 0) if rel is not None else None
                med_l = np.median(rel.values[il], 0) if rel is not None else None
            else:
                ih, il = codes == 1, codes == 0
                nh, nl = int(ih.sum()), int(il.sum())
                U = R[ih].sum(0) - nh * (nh + 1) / 2
                eff = 2 * U / (nh * nl) - 1                      # Cliff's delta (hi vs lo)
                diff = Ys[ih].mean(0) - Ys[il].mean(0)
                if rel is not None:
                    Rv = rel.values[pos]
                    med_h, med_l = np.median(Rv[ih], 0), np.median(Rv[il], 0)
        for j, f in enumerate(feats):
            row = dict(feature=f, test=test, contrast=contrast, lo=lo, hi=hi, F=r["F"][j],
                       effect=eff[j], p=r["p"][j], n=len(sub), covariates=",".join(covs))
            if paired:
                row["n_pairs"] = int(sub["patient"].nunique())
            if test == "pairwise":
                row.update(n_hi=nh, n_lo=nl, diff=diff[j])
                if rel is not None:
                    row.update(median_hi=med_h[j], median_lo=med_l[j])
            rows.append(row)
    return rows


# =============================================================== beta tests
def beta_tests(dists, mb, stratum, factor, levels, ordered, cov_candidates, perm_block, S, seeds,
               unit_meta=None, min_pairs=3):
    paired = factor == "timepoint"
    if unit_meta is None:
        names = stratum_samples(mb, stratum)
        pos = {s: i for i, s in enumerate(mb.index)}
        idx = np.array([pos[s] for s in names])
        m = mb.loc[names]
    else:
        m = unit_meta.reset_index(drop=True)
        idx = np.arange(len(m))
    lab = m[factor].values
    present = [l for l in levels if (lab == l).sum() >= 2]
    rows = []
    if len(present) < 2:
        return rows
    jobs = []
    if len(present) > 2:
        jobs.append(("omnibus", "all", None, None, present))
    for lo, hi in itertools.combinations(present, 2):
        jobs.append(("pairwise", f"{hi} vs {lo}", lo, hi, [lo, hi]))
    for metric, D in dists.items():
        Ds = D[np.ix_(idx, idx)]
        for kind, contrast, lo, hi, lv in jobs:
            if paired:
                mask = _patients_with_levels(m, factor, lv, len(lv) if kind == "pairwise" else 2)
            else:
                mask = np.isin(lab, lv)
            mm, DD = m[mask], Ds[np.ix_(mask, mask)]
            if mm[factor].nunique() < 2 or (paired and mm["patient"].nunique() < min_pairs):
                continue
            if stratum == "pooled" and "healthy_match" in mm:   # site covariate: a healthy sample counts as the
                hs = mm["site"].eq("healthy").values               # site it was matched to (else non-lesional)
                if hs.any():
                    mt = mm["healthy_match"].astype(object).fillna("").replace("", "non-lesional")
                    mm = mm.assign(site=mm["site"].where(~hs, mt))
            covs = [c for c in list(cov_candidates) + (["site"] if stratum == "pooled" else [])
                    if c in mm.columns and mm[c].fillna("n/a").nunique() > 1]
            if paired:
                covs = ["patient"] + [c for c in covs if c != "patient"]
            cov = cov_dummies(mm, covs)
            blocks = mm[perm_block].values if (perm_block in covs) else None
            codes = pd.Categorical(mm[factor], categories=lv).codes.astype(int)
            scheme = "within" if paired else "between"
            r = perm_test(DD, codes, len(lv), cov, scheme, mm["patient"].values, blocks,
                          S.n_perm, next(seeds), "dist")
            rows.append(dict(test="PERMANOVA", kind=kind, metric=metric, stratum=stratum,
                             contrast=contrast, lo=lo, hi=hi, n=len(mm), covariates=",".join(covs),
                             n_pairs=int(mm["patient"].nunique()) if paired else np.nan, **r))
            if S.permdisp and stratum != "pooled" and not paired:
                r = permdisp(DD, codes, len(lv), mm["patient"].values, blocks, S.n_perm, next(seeds))
                rows.append(dict(test="PERMDISP", kind=kind, metric=metric, stratum=stratum,
                                 contrast=contrast, lo=lo, hi=hi, n=len(mm), covariates="",
                                 n_pairs=np.nan, **r))
        if stratum == "pooled" and unit_meta is None:      # paired lesional vs non-lesional
            tp = (m["timepoint"].astype(object).where(m["timepoint"].notna(), "")
                  if "timepoint" in m.columns else pd.Series("", index=m.index))
            site_strata = (m["patient"].astype(str) + "|" + tp.astype(str)).values   # pair within patient AND timepoint
            site_jobs = [("site_overall", "lesional vs non-lesional (all)", np.ones(len(m), bool))]
            site_jobs += [("site_per_level", f"lesional vs non-lesional ({l})", lab == l) for l in levels]
            for kind, contrast, mask in site_jobs:
                mm, DD = m[mask], Ds[np.ix_(mask, mask)]
                if mm["site"].nunique() < 2 or mm["patient"].nunique() < 3:
                    continue
                codes = pd.Categorical(mm["site"], categories=["non-lesional", "lesional"]).codes.astype(int)
                r = perm_test(DD, codes, 2, np.zeros((len(mm), 0)), "within", site_strata[mask],
                              None, S.n_perm, next(seeds), "dist")
                rows.append(dict(test="PERMANOVA", kind=kind, metric=metric, stratum="pooled",
                                 contrast=contrast, lo=None, hi=None, n=len(mm), covariates="",
                                 n_pairs=np.nan, **r))
    return rows


def within_patient_frame(dists, mb, factor=None):
    """Mean lesional vs non-lesional distance. Pairs are formed WITHIN the same timepoint (never
    across). factor == 'timepoint' -> one row per patient-timepoint; otherwise one row per patient
    (mean over timepoints)."""
    pos = {s: i for i, s in enumerate(mb.index)}
    tp = (mb["timepoint"].astype(object).where(mb["timepoint"].notna(), "")
          if "timepoint" in mb.columns else pd.Series("", index=mb.index))
    metrics = list(dists)
    rows = []
    for (pat, t), g in mb.assign(_tp=tp.values).groupby(["patient", "_tp"], sort=True):
        L = [pos[s] for s in g.index[g["site"] == "lesional"]]
        N = [pos[s] for s in g.index[g["site"] == "non-lesional"]]
        if L and N:
            r = dict(patient=pat, severity=g["severity"].iloc[0], study=g["study"].iloc[0],
                     group=g["group"].iloc[0], skin_type=g["skin_type"].iloc[0],
                     body_site=" / ".join(g["body_site"].astype(str)),
                     samples=", ".join(g["sample"]), site="L vs NL",
                     timepoint=(t if t != "" else pd.NA))
            for metric in metrics:
                r[metric] = dists[metric][np.ix_(L, N)].mean()
            rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if factor == "timepoint":
        return df[df["timepoint"].notna()].reset_index(drop=True)
    agg = {c: "first" for c in df.columns if c not in metrics + ["patient"]}
    agg.update({mt: "mean" for mt in metrics})
    return df.groupby("patient", sort=True).agg(agg).reset_index().drop(columns=["timepoint"])


# =============================================================== block analysis
def analyze_block(block, md, mat, clr_mat, S, factor, levels_all, ordered, seeds, pseudo=None):
    mb = md.loc[block.names].copy()
    matb, clrb = mat.loc[block.names], clr_mat.loc[block.names]
    mb["shannon"] = shannon(matb)
    mb["richness"] = (matb > 0).sum(axis=1)
    dists = distance_matrices(matb, clrb)
    if pseudo is None:
        pseudo = float(mat.values[mat.values > 0].min() / 2)    # same CLR pseudo-count as clr_mat
    out = {k: [] for k in ("alpha", "da", "beta", "wdist", "counts")}
    taxa_union = set()

    for st in strata_for(set(mb["site"])):
        Ya, meta = unit_frame(mb, mb[["shannon", "richness"]], st, factor=factor)
        if Ya is None:
            continue
        levels = [l for l in levels_all if l in set(meta[factor])]
        for lv in levels:
            out["counts"].append(dict(stratum=st, level=lv, n_units=int((meta[factor] == lv).sum()),
                                      n_patients=int(meta.loc[meta[factor] == lv, "patient"].nunique())))
        if len(levels) < 2:
            continue
        for r in feature_tests(Ya, meta, factor, levels, ordered, block.cov_candidates,
                               block.perm_block, S, seeds):
            out["alpha"].append({**r, "stratum": st})
        sm = matb.loc[stratum_samples(mb, st, factor)]          # taxon filter within THIS stratum
        prev, mean = (sm > 0).mean(), sm.mean()
        taxa = [t for t in sm.columns[(prev >= S.min_prev) & (mean >= S.min_abund)]
                if t not in EXCLUDE_TAXA]
        taxa_union.update(taxa)
        if taxa:
            Yc, _ = unit_frame(mb, clrb[taxa], st, factor=factor)
            Yr, _ = unit_frame(mb, matb[taxa], st, factor=factor)
            for r in feature_tests(Yc, meta, factor, levels, ordered, block.cov_candidates,
                                   block.perm_block, S, seeds, rel=Yr):
                out["da"].append({**r, "stratum": st})
        if factor == "timepoint":
            Yb, meta_beta = unit_frame(mb, matb, st, factor=factor)
            Yb = Yb.loc[:, (Yb > 0).any(axis=0)]                   # taxa detected in this stratum
            beta_dists = distance_matrices(Yb, clr(Yb, pseudo))
            out["beta"] += beta_tests(beta_dists, mb, st, factor, levels, ordered,
                                      block.cov_candidates, block.perm_block, S, seeds,
                                      unit_meta=meta_beta)
        else:
            out["beta"] += beta_tests(dists, mb, st, factor, levels, ordered,
                                      block.cov_candidates, block.perm_block, S, seeds)

    if {"lesional", "non-lesional"} <= set(mb["site"]):
        wp = within_patient_frame(dists, mb, factor=factor)
        if len(wp):
            meta = wp.drop(columns=list(dists)).reset_index(drop=True)
            Yw = wp[list(dists)].reset_index(drop=True)
            levels = [l for l in levels_all if l in set(meta[factor])]
            for r in feature_tests(Yw, meta, factor, levels, ordered, block.cov_candidates,
                                   block.perm_block, S, seeds):
                out["wdist"].append({**r, "stratum": "within-patient L-NL"})
    frames = {}
    for k, rows in out.items():
        df = pd.DataFrame(rows)
        if len(df):
            df.insert(0, "block", block.label)
        frames[k] = df
    frames["taxa"] = sorted(taxa_union)
    return frames


def finalize(frames_list, S):
    """Concatenate block results and add q (within family, per block) and q_global."""
    tabs = {}
    for k in ("alpha", "da", "beta", "wdist", "counts"):
        parts = [f[k] for f in frames_list if k in f and len(f[k])]
        tabs[k] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    meth = S.correction
    if len(tabs["alpha"]):
        tabs["alpha"] = add_q(tabs["alpha"], ["block", "stratum"], meth)
        tabs["alpha"]["q_global"] = add_q(tabs["alpha"], [], meth)["q"].values
    if len(tabs["da"]):
        tabs["da"] = add_q(tabs["da"], ["block", "stratum", "contrast"], meth)
        tabs["da"]["q_global"] = add_q(tabs["da"], ["test"], meth)["q"].values
        tabs["da"]["log2fc"] = tabs["da"]["diff"] / np.log(2)
    if len(tabs["wdist"]):
        tabs["wdist"] = add_q(tabs["wdist"], ["block", "feature"], meth)
        tabs["wdist"]["q_global"] = add_q(tabs["wdist"], ["feature"], meth)["q"].values
    if len(tabs["beta"]):
        fam = {"omnibus": ["block", "test", "metric"], "pairwise": ["block", "test", "metric", "stratum"],
               "site_overall": ["block", "metric"], "site_per_level": ["block", "metric"],
               "local_between": ["block", "metric"], "local_within": ["block", "metric"]}
        parts = []
        for kind, cols in fam.items():
            sub = tabs["beta"][tabs["beta"]["kind"] == kind]
            if len(sub):
                sub = add_q(sub, cols, meth)
                sub["q_global"] = add_q(sub, ["test", "kind", "metric"], meth)["q"].values
                parts.append(sub)
        tabs["beta"] = pd.concat(parts, ignore_index=True)
    tabs["taxa"] = sorted({t for f in frames_list for t in f.get("taxa", [])})
    tabs["notes"] = []
    return tabs


def run_analysis(md, mat, clr_mat, S, factor="severity", levels_all=None, ordered=None,
                 progress=None, timepoint_order=None, pseudo=None):
    """Run every block. Returns (tables, blocks). `progress(fraction, text)` is optional.
    factor: 'severity' | 'study' | 'timepoint'. Defaults: levels from the data (pre/baseline first for
    timepoints); ordinal trend test only for severity, or for timepoints when `timepoint_order` is given."""
    fm = filter_md(md, S)
    if levels_all is None:
        if factor == "severity":
            levels_all = severity_levels(fm)
        elif factor == "study":
            levels_all = [s for s in S.studies if s in set(fm["study"])]
        elif factor == "timepoint":
            levels_all = timepoint_levels(fm, timepoint_order)
        else:
            raise ValueError(factor)
    if ordered is None:
        ordered = factor == "severity" or (factor == "timepoint" and bool(timepoint_order))
    notes = []
    if factor != "timepoint" and S.timepoint_mode == "average" and fm["timepoint"].notna().any():
        if (fm.groupby(["patient", "site"])["timepoint"].nunique() > 1).any():
            notes.append("Timepoints are averaged: each unit is the patient mean across the selected timepoints.")
    if factor != "timepoint" and S.timepoint_mode == "separate" and timeless_healthy(fm).any() \
            and fm.loc[~timeless_healthy(fm), "timepoint"].nunique() > 1:
        notes.append("Healthy volunteers sampled at one timepoint only are the reference at every timepoint.")
    blocks = make_blocks(md, S, factor)
    seeds = itertools.count(S.seed)
    frames = []
    for i, b in enumerate(blocks):
        if progress:
            progress(i / max(len(blocks), 1), f"Analysing block {i + 1}/{len(blocks)}: {b.label}")
        frames.append(analyze_block(b, md, mat, clr_mat, S, factor, levels_all, ordered, seeds, pseudo))
    if progress:
        progress(1.0, "Done")
    tabs = finalize(frames, S)
    tabs["notes"] = notes
    return tabs, blocks


def methods_text(S, factor="severity"):
    cov = []
    if S.study_mode == "adjust":
        cov.append("study (permutations restricted within study)")
    if S.group_mode == "adjust":
        cov.append("disease group")
    if S.skin_mode == "adjust":
        cov.append("skin type")
    sep = []
    if S.study_mode == "separate":
        sep.append("study")
    if S.group_mode == "separate":
        sep.append("disease group")
    if S.skin_mode == "stratify":
        sep.append("skin type")
    paired = ""
    if factor == "timepoint":
        paired = """
Timepoint comparisons are paired. Units are patient-timepoint means; a pairwise contrast uses only patients
observed at both timepoints (omnibus/trend: at least two). Patient enters as a fixed-effect covariate and
timepoint labels are permuted within patient. Effect sizes are paired: mean difference of CLR values
(log2 fold change) and the matched-pairs rank-biserial correlation. Beta diversity compares patient-timepoint
mean compositions with within-patient permutations; PERMDISP is not run for these repeated-measures comparisons.
"""
    return f"""STATISTICAL METHODS (auto-generated -- edit to match your final analysis)

Emu relative abundances were pooled per taxonomic rank and renormalised per sample. Taxa were
tested within each stratum when present in >= {S.min_prev:.0%} of that stratum's samples with mean relative
abundance >= {S.min_abund}. Differential abundance used CLR-transformed abundances (pseudo-count = half the
smallest non-zero abundance), computed separately for gut and for skin/nostril samples over the taxa detected
in that compartment. Alpha diversity: Shannon index and observed taxa.

Comparison factor: {factor}. Analyses were run separately within: {', '.join(sep) or 'nothing (one block)'};
adjusted for: {', '.join(cov) or 'no covariates'}. Strata: non-lesional, lesional, pooled (each patient
contributes the mean of their lesional and non-lesional sample; patients with both only), nostril and gut
(faecal, analysed on its own and never pooled with skin), as selected. One unit = one patient (patient-timepoint for timepoint comparisons), so repeated measures are never
treated as independent.

Univariate features: rank-transformed values, permutation F test (omnibus, ordinal trend, every pair),
{S.n_perm} permutations of patient-level labels, adjusted for covariates where applicable (equivalent to
Kruskal-Wallis / Spearman / Mann-Whitney without covariates). Effect sizes: Cliff's delta, log2 fold change (CLR).
Beta diversity: Bray-Curtis and Aitchison distances; PERMANOVA (marginal pseudo-F, distance-based linear model,
McArdle & Anderson 2001) with whole-patient permutations (within study when study is a covariate),
lesional vs non-lesional tested with permutations within patient (and timepoint); PERMDISP for dispersion
(not adjusted for covariates).
{paired}
Multiple testing: {S.correction.upper()} within families (taxa within block x stratum x contrast; alpha diversity
within block x stratum; pairwise PERMANOVA/PERMDISP within block x metric x stratum; omnibus tests within
block x metric). 'q_global' corrects across all blocks. Significant: q < {S.q_thr}. Results are exploratory;
abundances are compositional.
"""

