"""
datasets.py -- one data model for every modality (skin, gut, ceramides, energy metabolism).

A Dataset holds, with identical row index (one row per measured sample):
    mat     raw scale       relative abundance (compositional) or concentration (abundance; 0 = not detected)
    values  analysis scale  CLR per compartment (compositional) or ln(concentration) (abundance)
    md      the engine's standard metadata: study, sample, patient, site, group, severity,
            skin_type, timepoint, pasi, body_site
    derived optional per-sample summary indices (alpha diversity, ceramide class fractions, ratios)

Metadata (group, severity, PASI, global / local severity, skin type, body site) comes from the metadata file
only; see "the metadata file" below for how samples are matched to its rows.

How samples of different modalities are matched: study + patient + timepoint.
    patient    "<study>|<subject>". Ids are trimmed and '12.0' / '012' become '12'. If a table writes the
               subject differently from the metadata file (e.g. '53_AN' vs '53'), match_subjects()
               finds the rule that matches most patients and reports it.
    timepoint  every dataset keeps its own raw label; timepoint_map() proposes a canonical visit
               (baseline / mid / follow-up) per dataset, study and raw label, editable in Data setup.
    site       lesional / non-lesional / nostril / gut / blister / plasma ... is never averaged.

Zeros and missing values (concentration tables): a zero is read as "below the detection limit", not as a
real concentration of zero. By default such values are left out (NaN on the ln scale); set_imputation()
can fill them instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import skin_engine as E

MD_COLS = ["study", "sample", "patient", "site", "group", "severity", "skin_type", "timepoint", "pasi", "body_site",
           "lesion_rank", "lesion", "local_severity", "global_severity", "healthy_match"]
# location codes of a patient's second, less severe lesion in the concentration tables (first lesion: l1 / x1 / tl)
SECOND_LESION_CODES = {"x2", "le2", "l2", "lesional-2", "lesional2"}
SKIN_SITES = ("non-lesional", "lesional", "healthy", "nostril")
CANONICAL_TIMEPOINTS = list(E.TIMEPOINT_ORDER)            # baseline, mid, follow-up
# Authoritative study-specific timepoint label -> canonical visit mapping (overrides the auto-guess).
# Labels are the file's own codes. 1947: 2=pre, 61/62=mid, 99=post. 1806: 0=pre, 28=mid, 111/112=post. 2230: 1=pre.
TIMEPOINT_OVERRIDES = {
    ("1947", "2"): "baseline", ("1947", "61"): "mid", ("1947", "62"): "mid", ("1947", "99"): "follow-up",
    ("1806", "-14"): "screening", ("1806", "0"): "baseline", ("1806", "28"): "mid", ("1806", "111"): "follow-up", ("1806", "112"): "follow-up",
    ("2230", "1"): "baseline", ("2230", "2"): "follow-up",
}


def _override_timepoint(study, label) -> str | None:
    return TIMEPOINT_OVERRIDES.get((str(study), str(label).strip()))


NO_LABEL = "(none)"
NOT_APPLICABLE = {"nvt", "nvt.", "n.v.t.", "na", "n/a", "n.a.", "-", "none", "nan", ""}
IMPUTATIONS = {
    "none": "leave out (no imputation)",
    "half-min": "half of the smallest measured value (below detection limit)",
    "knn": "k-nearest neighbours (value missing by chance)",
}

# column aliases of the concentration tables (lower-case, spaces -> '_'). First match wins.
COLUMN_ALIASES = {
    "study": ["study", "study_id", "studynr", "protocol"],
    "subject": ["subject", "subj_id", "subject_id", "subjid", "patient", "patient_id", "participant", "subj"],
    "timepoint": ["timepoint", "sampling_time", "occ", "occasion", "visit", "time", "day", "week", "timepoint_code"],
    "site": ["location", "lesion_status", "lesion", "site", "matrix", "compartment", "specimen", "site_code", "lesion_site",
             "sample_source"],
    "batch": ["batch", "plate", "run"],
    "value": ["value", "concentration", "conc", "amount", "result", "intensity", "area", "relative_abundance"],
    "_ignore": ["lcab_id", "sample_id", "sample", "sample_nr", "id", "nr"],
}
# Metadata columns a concentration table may also have. They are NOT used: group, severity, PASI, global and local
# severity, skin type and body site come from the metadata file only (see "the metadata file" below).
METADATA_COLUMNS = ["stage", "severity", "group", "diagnosis", "disease", "cohort", "population", "pasi",
                    "global_severity", "local_severity", "lesion_rank", "region", "skin_type", "region_location",
                    "body_site"]
# columns that can hold a sample id, to match a table's samples to the metadata file's sample_id (first found wins)
ID_COLUMNS = ["sample_id", "lcab_id", "sample", "sample_nr", "id", "nr"]


@dataclass
class Dataset:
    name: str                       # 'skin' | 'gut' | 'ceramides' | 'energy'
    label: str
    kind: str                       # 'compositional' | 'abundance'
    mat: pd.DataFrame
    values: pd.DataFrame
    md: pd.DataFrame
    feature_label: str = "feature"
    pseudo: float | None = None
    derived: pd.DataFrame | None = None
    feature_info: pd.DataFrame | None = None
    notes: list = field(default_factory=list)
    raw_timepoint: pd.Series | None = None      # the file's own timepoint label per sample
    site_titles: dict = field(default_factory=dict)   # display name per site (e.g. lesional -> blister lesional)
    patient_site: dict = field(default_factory=dict)  # site of a PATIENT's sample at a healthy-reference site
    healthy_sites: tuple = ()                          # sites whose healthy samples become site 'healthy' 
    log_observed: pd.DataFrame | None = None    # abundance: ln(concentration), NaN where zero / missing
    imputation: str = "none"
    ratio_defs: list = field(default_factory=list)
    sample_ids: pd.Series | None = None         # the table's own sample id (sample_id / LCAB_ID ...) per sample
    metadata_match: pd.DataFrame | None = None  # how every sample was matched to the metadata file

    @property
    def sites(self):
        present = list(pd.unique(self.md["site"]))
        return [s for s in E.SITES if s in present] + sorted(s for s in present if s not in E.SITES)

    def site_title(self, site) -> str:
        return self.site_titles.get(site, site)

    def set_imputation(self, method: str):
        """Abundance data only: how zero / missing values enter the analysis ('none' leaves them out)."""
        if self.kind != "abundance" or method == self.imputation:
            return
        self.values = impute(self.log_observed, self.md, method)
        self.imputation = method


# =============================================================== keys and numbers
def norm_subject(value) -> str:
    s = re.sub(r"\.0+$", "", str(value).strip())           # 12.0 (Excel) -> 12
    if s.isdigit():
        s = s.lstrip("0") or "0"                            # 012 -> 12
    return s.upper()


def norm_study(value) -> str:
    m = re.search(r"(\d{4})", str(value))                   # 'CHDR1806' -> '1806'
    return m.group(1) if m else str(value).strip()


def norm_timepoint(series: pd.Series) -> pd.Series:
    t = series.astype("string").str.strip().str.lower().str.replace(r"\.0+$", "", regex=True)
    return t.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "<na>": pd.NA})


def _text(series: pd.Series) -> pd.Series:
    """Plain Python strings, '' for missing (pandas string columns keep NaN through astype(str))."""
    return series.astype(object).map(lambda v: "" if pd.isna(v) else str(v).strip())


def to_number(series: pd.Series) -> pd.Series:
    """Numbers written with a decimal comma or a decimal point ('1,100' -> 1.1; '1.234,5' -> 1234.5)."""
    def fix(v):
        v = re.sub(r"\s", "", v)
        if "," in v and "." in v:
            return v.replace(".", "").replace(",", ".") if v.rfind(",") > v.rfind(".") else v.replace(",", "")
        return v.replace(",", ".")
    return pd.to_numeric(_text(series).map(fix), errors="coerce")


# =============================================================== microbiome
def microbiome_datasets(mat: pd.DataFrame, md: pd.DataFrame, notes=()) -> dict:
    """Split the load_studies() output into a skin (incl. nostril) and a gut Dataset."""
    md = md.copy()
    parts = md["patient"].astype(str).str.split("|", n=1, expand=True)
    md["patient"] = parts[0].map(norm_study) + "|" + parts[1].map(norm_subject)
    raw_tp = norm_timepoint(md["timepoint"])
    pseudo = float(mat.values[mat.values > 0].min() / 2)
    clr_all = E.clr_by_compartment(mat, md, pseudo)          # once, on everything: independent of filters
    out = {}
    for name, label, mask in (("skin", "Skin microbiome", md["site"].isin(SKIN_SITES)),
                              ("gut", "Gut microbiome", md["site"].eq("gut"))):
        idx = md.index[mask.values]
        if len(idx) == 0:
            continue
        sub = mat.loc[idx]
        keep = sub.columns[(sub > 0).any(axis=0)]
        derived = pd.DataFrame({"Shannon": E.shannon(sub), "Observed taxa": (sub > 0).sum(axis=1).astype(float)})
        out[name] = Dataset(name, label, "compositional", sub[keep], clr_all.loc[idx, keep], md.loc[idx, MD_COLS].copy(),
                            feature_label="taxon", pseudo=pseudo, derived=derived,
                            notes=list(notes) if name == "skin" else [], raw_timepoint=raw_tp.loc[idx])
    return out


# =============================================================== concentration tables
def tidy_table(raw: pd.DataFrame, feature_aliases, name: str, default_study=None, default_site=None):
    """Standardise a long or wide concentration table. Returns (tidy long DataFrame, notes).
    tidy columns: study, subject, timepoint, site, feature, value (+ group, severity, pasi ... when present)."""
    notes = []
    df = raw.copy()
    df.columns = [re.sub(r"[\s\-]+", "_", str(c).strip().lower()) for c in df.columns]
    original = dict(zip(df.columns, [str(c).strip() for c in raw.columns]))
    ren = {}
    for target, aliases in {**COLUMN_ALIASES, "feature": list(feature_aliases)}.items():
        for a in aliases:
            if a in df.columns and a not in ren and (target == "_ignore" or target not in ren.values()):
                ren[a] = target if target != "_ignore" else f"_ignore_{a}"
                if target != "_ignore":
                    break
    for c in METADATA_COLUMNS:
        if c in df.columns and c not in ren:
            ren[c] = f"_meta_{c}"
    df = df.rename(columns=ren)
    skipped = [original.get(c, c) for c, t in ren.items() if t.startswith("_meta_")]
    if skipped:
        notes.append(f"[{name}] metadata column(s) in this table are not used (metadata comes from the metadata file "
                     f"only): {skipped}")
    if "subject" not in df:
        raise ValueError(f"[{name}] no subject column found. Columns: {list(raw.columns)}")
    if "value" not in df or "feature" not in df:               # wide table -> long
        ids = [c for c in df.columns if c in COLUMN_ALIASES or c.startswith(("_ignore_", "_meta_"))]
        feats = [c for c in df.columns if c not in ids and to_number(df[c]).notna().any()]
        if not feats:
            raise ValueError(f"[{name}] need a long table (feature + value columns) or a wide table with "
                             f"numeric feature columns. Columns: {list(raw.columns)}")
        df = df.melt(id_vars=ids, value_vars=feats, var_name="feature", value_name="value")
        df["feature"] = df["feature"].map(lambda c: original.get(c, c))
    text = _text(df["value"])
    df["value"] = to_number(df["value"])
    odd = df["value"].isna() & ~text.str.lower().isin({"", "nan", "none", "<na>"})
    if odd.any():
        notes.append(f"[{name}] {int(odd.sum())} value(s) are not numbers (e.g. {sorted(set(text[odd]))[:4]}) and "
                     f"count as missing")
    df["feature"] = _text(df["feature"])
    if "study" not in df:
        if default_study is None:
            raise ValueError(f"[{name}] no study column found and no default study given")
        df["study"] = default_study
    df["study"] = df["study"].map(norm_study)
    df["subject"] = df["subject"].map(norm_subject)
    df["timepoint"] = norm_timepoint(df["timepoint"]) if "timepoint" in df else pd.NA
    id_col = next((f"_ignore_{c}" for c in ID_COLUMNS if f"_ignore_{c}" in df), None)
    df["_sample_id"] = _text(df[id_col]) if id_col else ""
    if "site" in df:
        raw_site = _text(df["site"]).str.lower().str.replace(r"[ _]", "-", regex=True)
        healthy = raw_site.isin(E.HEALTHY_LABELS)
        site = raw_site.replace({c: "lesional" for c in SECOND_LESION_CODES}).replace(E.SITE_ALIASES)
        site = site.replace({"rnos": "nostril", "nan": default_site or name, "": default_site or name})
        df["site"] = site.mask(healthy, "healthy")
        df["_healthy_site"] = healthy
        # two lesions of one patient are written as two location codes (e.g. X1 and X2): keep them apart
        df["_second_lesion"] = raw_site.isin(SECOND_LESION_CODES)
        if "lesion" not in df:
            df["lesion"] = np.where(df["_second_lesion"], "second lesion (" + raw_site.str.upper() + ")", "")
    else:
        df["site"] = default_site or name
        df["_healthy_site"] = False
    return df, notes


def abundance_dataset(name, label, tidy, notes, feature_label) -> Dataset:
    """Build a Dataset from a tidy long concentration table (see tidy_table)."""
    notes = list(notes)
    tidy = tidy.copy()
    tidy["patient"] = tidy["study"] + "|" + tidy["subject"]
    tp = tidy["timepoint"].astype(object).where(tidy["timepoint"].notna(), "")
    if "lesion" in tidy.columns:
        les = tidy["lesion"].astype(object).where(tidy["lesion"].notna(), "").astype(str)
    else:
        les = pd.Series("", index=tidy.index)
    tidy["uid"] = tidy["patient"] + "|" + tidy["site"].astype(str) + "|" + tp.astype(str) + "|" + les
    dup = int(tidy.dropna(subset=["value"]).duplicated(["uid", "feature"]).sum())
    if dup:
        notes.append(f"[{name}] {dup} repeated measurement(s) of the same sample and {feature_label} were averaged")
    mat = tidy.pivot_table(index="uid", columns="feature", values="value", aggfunc="mean")
    n_zero, n_blank, n_neg = int(mat.eq(0).values.sum()), int(mat.isna().values.sum()), int(mat.lt(0).values.sum())
    observed = mat.gt(0)
    keep = observed.any(axis=0)
    if (~keep).any():
        notes.append(f"[{name}] dropped {int((~keep).sum())} {feature_label}(s) without any value above zero")
    mat, observed = mat.loc[:, keep], observed.loc[:, keep]
    if n_zero or n_blank or n_neg:
        notes.append(f"[{name}] {n_zero} zero, {n_blank} missing and {n_neg} negative value(s) "
                     f"({(n_zero + n_blank + n_neg) / max(observed.size, 1):.1%} of all values) are left out of the "
                     f"analysis of that {feature_label} unless imputation is switched on")
    log_obs = np.log(mat.where(observed))

    first = tidy.drop_duplicates("uid").set_index("uid").loc[mat.index]
    md = pd.DataFrame(index=mat.index)
    md["study"] = first["study"].values
    md["sample"] = [u.split("|", 1)[1] for u in mat.index]
    md["patient"] = first["patient"].values
    md["site"] = first["site"].astype(str).values
    # group, severity, PASI, skin type, body site and the severity scores come from the metadata file only
    # (filled by harmonise / apply_timepoint_map); a healthy-volunteer site label only sets the site
    for c in ("group", "severity", "skin_type", "body_site"):
        md[c] = pd.array([np.nan] * len(md), dtype=object)
    md["timepoint"] = pd.array(first["timepoint"].values, dtype="string")
    for c in ("pasi", "local_severity", "global_severity"):
        md[c] = np.nan
    md["lesion"] = pd.array(first["lesion"].values, dtype="string") if "lesion" in first else pd.array([""] * len(first), dtype="string")
    md["healthy_match"] = ""
    second = (first["_second_lesion"].eq(True).values if "_second_lesion" in first
              else np.zeros(len(first), bool))
    md["lesion_rank"] = np.where(md["site"].eq("lesional"), np.where(second, 2.0, 1.0), np.nan)
    md["lesion_rank"] = rank_lesions(md)
    n_multi = int((md["lesion_rank"] > 1).sum())
    if n_multi:
        notes.append(f"[{name}] {md.loc[md['lesion_rank'] > 1, 'patient'].nunique()} patient(s) have more than one "
                     f"lesional sample at a timepoint (lesion codes); lesion 1 is matched to the most severe lesion "
                     f"of the metadata file and lesion 2 to the less severe one")
    ids = first["_sample_id"] if "_sample_id" in first else pd.Series("", index=first.index)
    return Dataset(name, label, "abundance", mat.where(observed, 0.0), log_obs.copy(), md[MD_COLS],
                   feature_label=feature_label, notes=notes, raw_timepoint=md["timepoint"].copy(),
                   log_observed=log_obs, sample_ids=pd.Series(_text(ids).values, index=md.index))


def rank_lesions(md: pd.DataFrame) -> pd.Series:
    """lesion_rank (1 = most severe) of the lesional samples within patient and timepoint: by local severity
    (highest first) when the lesions have one, otherwise by the rank already given. NaN for other sites."""
    rank = pd.to_numeric(md["lesion_rank"], errors="coerce").copy()
    les = md["site"].eq("lesional")
    if not les.any():
        return rank
    tp = md["timepoint"].astype(object).where(md["timepoint"].notna(), "").astype(str)
    sev = pd.to_numeric(md["local_severity"], errors="coerce")
    for _, idx in md[les].groupby([md.loc[les, "patient"].astype(str).values, tp[les].values]).groups.items():
        if len(idx) < 2:
            rank.loc[idx] = 1.0
        elif sev.loc[idx].notna().all() and sev.loc[idx].nunique() > 1:
            rank.loc[idx] = sev.loc[idx].rank(ascending=False, method="min")
        else:
            rank.loc[idx] = rank.loc[idx].fillna(1.0).rank(method="min")
    return rank


# =============================================================== imputation (optional)
def nan_euclidean(Z: np.ndarray) -> np.ndarray:
    """Euclidean distance using the coordinates both rows have, scaled up to the full number of coordinates."""
    ok = ~np.isnan(Z)
    X = np.where(ok, Z, 0.0)
    sq = (X ** 2) @ ok.T + ok @ (X ** 2).T - 2 * X @ X.T
    n = ok.astype(float) @ ok.T
    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.sqrt(np.clip(sq, 0, None) * Z.shape[1] / n)
    D[n == 0] = np.nan
    np.fill_diagonal(D, 0.0)
    return D


def impute(log_obs: pd.DataFrame, md: pd.DataFrame, method: str, k: int = 5) -> pd.DataFrame:
    """none: unchanged. half-min: ln(smallest measured value / 2) per feature, for values below the detection
    limit. knn: mean of the k most similar samples of the same site (z-scored ln scale), for values missing
    by chance; it is not suited to values that are missing because they are low."""
    if method == "none":
        return log_obs.copy()
    if method == "half-min":
        return log_obs.fillna(log_obs.min() - np.log(2))
    if method != "knn":
        raise ValueError(method)
    out = log_obs.copy()
    for site in pd.unique(md["site"]):
        rows = md.index[md["site"] == site]
        X = log_obs.loc[rows]
        mu, sd = X.mean(), X.std(ddof=0).replace(0, 1).fillna(1)
        Z = ((X - mu) / sd).values
        D = nan_euclidean(Z)
        np.fill_diagonal(D, np.inf)
        D = np.where(np.isnan(D), np.inf, D)
        filled = Z.copy()
        for i, j in zip(*np.where(np.isnan(Z))):
            cand = np.where(~np.isnan(Z[:, j]) & np.isfinite(D[i]))[0]
            near = cand[np.argsort(D[i, cand])[:k]]
            filled[i, j] = Z[near, j].mean() if len(near) else 0.0
        out.loc[rows] = filled * sd.values + mu.values
    return out.fillna(log_obs.mean())



# =============================================================== the metadata file
# The metadata file (meta_data_all_complete.csv) is the ONLY source of sample metadata: group, severity (stage),
# PASI, global severity, local severity, skin type and body site. Data files only identify their samples (sample
# id, or study + subject + site + timepoint + lesion code). A sample is matched to a metadata row
#   1. by sample id: the table's sample id (microbiome: sample_id; concentration tables: sample_id / LCAB_ID ...)
#      equals the metadata sample_id, or else
#   2. by study + patient + site + timepoint (+ lesion number for a patient with several lesions at a visit).
# Patient-level fields (group, severity, PASI) need only the patient; the global severity needs the visit
# (study + patient + timepoint); local severity, skin type and body site need the matched row.
METADATA_LABEL = "Metadata file"
MICROBIOME_LABELS = ("Skin microbiome", "Gut microbiome")


@dataclass
class MetadataReference:
    path: str
    rows: pd.DataFrame                  # one row per metadata row (standardised, see load_metadata_reference)
    patients: pd.DataFrame              # index 'study|subject': group, severity, pasi
    notes: list = field(default_factory=list)
    label: str = METADATA_LABEL

    @property
    def subjects(self) -> dict:
        out = {}
        for p in self.patients.index:
            study, subj = p.split("|", 1)
            out.setdefault(study, set()).add(subj)
        return out


def read_metadata_file(path) -> pd.DataFrame:
    """The metadata file as text, lower-case column names (csv like load_studies, or Excel)."""
    p = str(path)
    m = pd.read_excel(p, dtype=str) if p.lower().endswith((".xlsx", ".xls")) else pd.read_csv(p, dtype=str)
    m.columns = [str(c).strip().lower() for c in m.columns]
    return m


def load_metadata_reference(path, pasi_cutoffs=(10, 20)) -> MetadataReference:
    """Standardise every row of the metadata file (all studies and sites), with the same rules as the microbiome
    loader: one group per subject (skin rows first, 'healthy' wins), stage per patient, PASI = the highest global
    severity of a psoriasis patient (severity mild / moderate / severe from it when there is no stage), the lesion
    number from local_severity (1 = most severe lesion of the visit)."""
    import load_studies as L                       # same lesion and score rules as the microbiome loader
    m = read_metadata_file(path)
    missing = {"study", "subject"} - set(m.columns)
    if missing:
        raise ValueError(f"metadata file {path} has no column(s) {sorted(missing)}")
    notes = []
    get = lambda c: m[c] if c in m else pd.Series(np.nan, index=m.index, dtype=object)
    sample = _text(get("sample_id"))
    study = _text(m["study"]).map(lambda v: norm_study(v) if v else "")
    subj_raw = _text(m["subject"])
    subj_raw = subj_raw.where(subj_raw.ne(""), sample)
    keep = study.ne("") & subj_raw.ne("")
    if (~keep).any():
        notes.append(f"[metadata] {int((~keep).sum())} row(s) without study or subject are not used")
    m, sample, study, subj_raw = m[keep].copy(), sample[keep], study[keep], subj_raw[keep]
    subject = subj_raw.map(norm_subject)
    patient = study + "|" + subject

    compartment = _text(m["site"]).str.lower() if "site" in m else pd.Series("", index=m.index)
    status = _text(get("lesion_status")).str.lower().str.replace(r"[ _]", "-", regex=True)
    status = status.replace({"rnos": "nostril", "nonlesional": "non-lesional", "non-lesion": "non-lesional"})
    status = status.replace(E.SITE_ALIASES)
    if "sample_source" in m:          # e.g. energy metadata: 'blister perilesional' / 'blister non-lesional' / 'plasma'
        # kept apart from the skin sites ('blister-lesional', 'blister-non-lesional', 'blister', 'plasma'): a blister
        # is never a second skin lesion of the visit, and skin samples are never matched to a blister row
        src = _text(m["sample_source"]).str.lower().str.replace(r"[ _]", "-", regex=True)
        kind = pd.Series(np.where(src.str.contains("plasma|serum"), "plasma",
                                  np.where(src.str.contains("blister"), "blister", "")), index=m.index)
        sub = src.str.replace("blister", "", regex=False).str.strip("-").replace(E.SITE_ALIASES)
        label = kind + np.where(kind.eq("blister") & sub.isin(["lesional", "non-lesional"]), "-" + sub, "")
        status = status.where(status.ne("") | kind.eq(""), label)
    site = status.where(status.ne(""), compartment).mask(compartment.eq("gut"), "gut")
    site = site.mask(site.isin(E.HEALTHY_LABELS), "healthy")
    raw_tp = norm_timepoint(get("timepoint"))
    rows_for_rank = m.assign(timepoint=raw_tp.astype(object).where(raw_tp.notna(), "").astype(str))
    site, lesion_rank, _, rank_notes = L.rank_lesions(rows_for_rank, patient, site.astype(object), "metadata")
    notes += [n.replace("load_studies.py", "load_studies.py (metadata file)") for n in rank_notes]

    # patient level: group (one per subject; skin rows first, 'healthy' wins), severity, PASI
    def one_group(vals):
        vals = [v for v in vals if v and v.lower() != "nan"]
        if not vals:
            return np.nan
        if any(v.lower() in E.HEALTHY_LABELS for v in vals):
            return E.HEALTHY
        return vals[0]
    by_patient = pd.DataFrame({"patient": patient, "group": _text(get("group")),
                               "skin_first": compartment.ne("skin").astype(int),
                               "stage": _text(get("stage")).str.lower().replace({"none": "", "unknown": "", "nan": ""})})
    by_patient = by_patient.sort_values("skin_first", kind="stable").groupby("patient")
    group = by_patient["group"].agg(lambda g: one_group(g.tolist()))
    stages = by_patient["stage"].agg(lambda g: sorted({v for v in g if v}))
    several = stages[stages.map(len) > 1]
    if len(several):
        notes.append(f"[metadata] {len(several)} patient(s) have more than one stage (e.g. {several.index[0]}: "
                     f"{several.iloc[0]}); the first one in the file is used")
    severity = by_patient["stage"].agg(lambda g: next((v for v in g if v), np.nan)).astype(object)
    global_score = to_number(get("global_severity"))
    is_psoriasis = group.astype(str).str.lower().eq("psoriasis")
    pasi = global_score.groupby(patient.values).max().reindex(group.index).where(is_psoriasis)
    need = severity.isna() & pasi.notna()
    c1, c2 = pasi_cutoffs
    severity[need] = pd.cut(pasi[need], [-np.inf, c1, c2, np.inf], labels=["mild", "moderate", "severe"],
                            right=False).astype(object)
    healthy = group.eq(E.HEALTHY)
    severity = severity.mask(healthy, E.HEALTHY)
    patients = pd.DataFrame({"group": group.where(group.notna(), "unspecified"),
                             "severity": severity.where(severity.notna(), "unspecified"), "pasi": pasi})

    local = L.local_severity_values(m, site)
    skin_type = _text(get("region")).str.lower().map(E.SKIN_ALIASES).where(compartment.ne("gut"))
    body_site = _text(get("region_location")).where(_text(get("region_location")).ne(""),
                                                    np.where(compartment.eq("gut"), "faecal", ""))
    rows = pd.DataFrame({"sample_id": sample, "study": study, "patient": patient, "site": site.astype(str),
                         "compartment": compartment, "raw_timepoint": raw_tp, "lesion_rank": lesion_rank,
                         "local_severity": local, "global_severity": global_score, "skin_type": skin_type,
                         "body_site": body_site}).reset_index(drop=True)
    rows["group"] = rows["patient"].map(patients["group"])
    dup = rows["sample_id"].ne("") & rows.duplicated(["study", "sample_id"], keep=False)
    if dup.any():
        notes.append(f"[metadata] {int(dup.sum())} rows share a sample_id within a study (e.g. "
                     f"{rows.loc[dup, 'sample_id'].iloc[0]}); those samples are matched on patient, site and timepoint")
    rows["id_unique"] = rows["sample_id"].ne("") & ~dup
    notes.insert(0, f"[metadata] {path}: {len(rows)} rows, {len(patients)} patients in studies "
                    f"{', '.join(sorted(rows['study'].unique()))}. Group, severity, PASI, global and local severity, "
                    f"skin type and body site of every dataset come from this file only.")
    return MetadataReference(str(path), rows, patients, notes)


def apply_patient_metadata(datasets: dict, ref: MetadataReference | None) -> list:
    """group, severity and PASI of every sample from the metadata file (by patient). Patients not in the file:
    'unspecified'."""
    notes = []
    for name, ds in datasets.items():
        md = ds.md
        pat = (ref.patients.reindex(md["patient"].astype(str)) if ref is not None
               else pd.DataFrame(index=md["patient"].astype(str), columns=["group", "severity", "pasi"]))
        md["group"] = pd.array(pat["group"].where(pat["group"].notna(), "unspecified").values, dtype=object)
        md["severity"] = pd.array(pat["severity"].where(pat["severity"].notna(), "unspecified").values, dtype=object)
        md["pasi"] = pd.to_numeric(pat["pasi"], errors="coerce").values
        missing = pat["group"].isna().values
        if missing.any():
            notes.append(f"[{name}] {md.loc[missing, 'patient'].nunique()} patient(s) are not in the metadata file "
                         f"(e.g. {sorted(md.loc[missing, 'patient'].unique())[:5]}): group and severity 'unspecified', "
                         f"no scores")
    return notes


def _ref_timepoints(ref: MetadataReference, tab: pd.DataFrame) -> pd.Series:
    """Canonical timepoint of every metadata row: the mapping of the microbiome datasets (same file) for that
    study and label, else the 'Metadata file' rows of the timepoint table."""
    lookup = {}
    for r in tab.to_dict("records"):
        if r["dataset"] in MICROBIOME_LABELS + (METADATA_LABEL,):
            lookup.setdefault((str(r["study"]), str(r["label in file"])), r["timepoint"])
    raw = ref.rows["raw_timepoint"].astype(object).where(ref.rows["raw_timepoint"].notna(), NO_LABEL).astype(str)
    return pd.Series([lookup.get((s, l)) or _override_timepoint(s, l) or "baseline"
                      for s, l in zip(ref.rows["study"], raw)], index=ref.rows.index)


def _visit_global_severity(rows: pd.DataFrame) -> tuple[pd.Series, str | None]:
    """Global severity per visit (study, patient, canonical timepoint) from the metadata rows: the value when every
    row of the visit that has one agrees, else missing. Returns (Series, note about disagreeing visits)."""
    known = rows[pd.to_numeric(rows["global_severity"], errors="coerce").notna()].copy()
    known["v"] = pd.to_numeric(known["global_severity"], errors="coerce")
    stats = known.groupby(["study", "patient", "timepoint"])["v"].agg(["min", "max"])
    agree = np.isclose(stats["min"], stats["max"])
    visit, conflict = stats.loc[agree, "max"], stats.loc[~agree]
    if not len(conflict):
        return visit, None
    raw = known["raw_timepoint"].astype(object).where(known["raw_timepoint"].notna(), NO_LABEL).astype(str)
    known = known.assign(raw=raw.values)
    hit = known.merge(conflict.reset_index()[["study", "patient", "timepoint"]])
    several_codes = (hit.groupby(["study", "patient", "timepoint"])["raw"].nunique() > 1)
    ex = []
    for (sd, pat, t), g in list(hit.groupby(["study", "patient", "timepoint"]))[:3]:
        parts = g.groupby(["raw", "site"])["v"].agg(lambda v: "/".join(f"{x:g}" for x in sorted(set(v))))
        ex.append(f"{pat.split('|')[-1]} at {t}: " + ", ".join(f"{v} ({site}, file timepoint '{r}')"
                                                           for (r, site), v in parts.items()))
    why = []
    if several_codes.any():
        why.append(f"in {int(several_codes.sum())} of them, several timepoint codes of the metadata file are mapped "
                   f"to the same timepoint (check 'Timepoint labels')")
    if (~several_codes).any():
        why.append(f"in {int((~several_codes).sum())}, the rows of one timepoint code disagree in the file itself")
    note = (f"[metadata] {len(conflict)} visit(s) (same study, patient and timepoint) have different global severity "
            f"values in different rows: {'; '.join(why)}. A sample matched to a row keeps that row's value; other "
            f"samples of these visits have no global severity (nothing is averaged or maximised). E.g. "
            + "; ".join(ex))
    return visit, note


def apply_sample_metadata(datasets: dict, ref: MetadataReference | None, tab: pd.DataFrame | None) -> list:
    """global severity (per visit), local severity, skin type and body site of every sample from the metadata
    file. Run after the timepoint mapping: the timepoint of a match is the canonical one."""
    notes = []
    if ref is None:
        for ds in datasets.values():
            for c in ("local_severity", "global_severity", "skin_type", "body_site"):
                ds.md[c] = np.nan
            ds.metadata_match = None
        return ["[metadata] no metadata file: group, severity and scores are missing for every dataset"]
    rows = ref.rows.copy()
    rows["timepoint"] = _ref_timepoints(ref, tab) if tab is not None else "baseline"
    # global severity = the score of ONE visit (study + patient + timepoint). A sample matched to a metadata row
    # keeps that row's own value; otherwise it takes the visit's value, but only when all rows of the visit agree.
    # Nothing is taken from another timepoint, and nothing is averaged or maximised.
    visit, conflict_note = _visit_global_severity(rows)
    all_visits = set(zip(rows["study"], rows["patient"], rows["timepoint"]))
    if conflict_note:
        notes.append(conflict_note)
    by_id = rows[rows["id_unique"]].set_index(["study", "sample_id"], drop=False)
    by_key = {k: g for k, g in rows.groupby(["study", "patient", "site", "timepoint"])}
    for name, ds in datasets.items():
        md = ds.md
        if getattr(ds, "own_lesion_rank", None) is None:     # the dataset's own lesion numbers (codes X1 / X2)
            ds.own_lesion_rank = pd.to_numeric(md["lesion_rank"], errors="coerce").copy()
        own_rank = ds.own_lesion_rank.reindex(md.index)
        tp = md["timepoint"].astype(object).where(md["timepoint"].notna(), "").astype(str)
        if name in ("skin", "gut"):
            ids = md["sample"].astype(str)
        else:
            ids = (ds.sample_ids.reindex(md.index).fillna("").astype(str) if ds.sample_ids is not None
                   else pd.Series("", index=md.index))
        lesional = md["site"].eq("lesional")
        n_les = lesional.groupby([md["patient"].astype(str).values, tp.values]).transform("sum")
        out = pd.DataFrame(index=md.index, columns=["matched by", "metadata sample", "local_severity", "skin_type",
                                                    "body_site"], dtype=object)
        out["matched by"] = "patient only"
        ambiguous = no_lesion = 0
        own_global = {}                                    # the matched row's own global severity
        for i in md.index:
            study, patient, site = str(md.at[i, "study"]), str(md.at[i, "patient"]), str(md.at[i, "site"])
            row, how = None, None

            def key_match(exclude=""):
                nonlocal ambiguous, no_lesion
                row, how = None, None
                cand = by_key.get((study, patient, site, tp[i]))
                if cand is not None and exclude:
                    cand = cand[cand["sample_id"].ne(exclude)]
                    cand = cand if len(cand) else None
                if cand is not None and len(cand) == 1 and site == "lesional" and own_rank[i] > 1:
                    no_lesion += 1                         # lesion 2 of the table; the metadata file has one lesion
                elif cand is not None and len(cand) == 1:
                    row, how = cand.iloc[0], "patient + site + timepoint"
                elif cand is not None and site == "lesional":
                    # several lesions of the patient at this visit: lesion number 1 = most severe, 2 = less severe
                    hit = cand[pd.to_numeric(cand["lesion_rank"], errors="coerce") == own_rank[i]]
                    if len(hit) == 1 and (n_les[i] > 1 or own_rank[i] > 1):
                        row, how = hit.iloc[0], "patient + site + timepoint + lesion number"
                    else:
                        ambiguous += 1
                elif cand is not None:                     # several rows of one site: use what they agree on
                    same = cand[["skin_type", "body_site"]].nunique(dropna=False) <= 1
                    row = pd.Series({"sample_id": "", "local_severity": np.nan,
                                     "skin_type": cand["skin_type"].iloc[0] if same["skin_type"] else np.nan,
                                     "body_site": cand["body_site"].iloc[0] if same["body_site"] else np.nan})
                    how = "patient + site + timepoint (several rows)"
                return row, how

            if ids[i] and (study, ids[i]) in by_id.index:
                row, how = by_id.loc[(study, ids[i])], "sample id"
                # e.g. a blister's own metadata row (sample source, visit) has no lesion score, skin type or body
                # site: the skin row of that site and visit has them
                blank = lambda v: v is None or v == "" or (not isinstance(v, str) and pd.isna(v))
                fields = [c for c in ("local_severity", "skin_type", "body_site") if blank(row.get(c))
                          and (c != "local_severity" or site == "lesional")]
                if fields and str(row.get("site")) != site:                # the row is not a skin row itself
                    extra, how2 = key_match(exclude=row["sample_id"])
                    got = [c for c in fields if extra is not None and not blank(extra.get(c))]
                    if got:
                        row = row.copy()
                        for c in got:
                            row[c] = extra.get(c)
                        how = f"sample id; {', '.join(c.replace('_', ' ') for c in got)}: {how2}"
            else:
                row, how = key_match()
            if row is not None:
                out.loc[i, ["matched by", "metadata sample", "local_severity", "skin_type", "body_site"]] = [
                    how, row["sample_id"], row["local_severity"], row["skin_type"], row["body_site"]]
                own_global[i] = row.get("global_severity", np.nan)
            elif (study, patient, tp[i]) in all_visits:
                out.at[i, "matched by"] = "patient + timepoint"
            if patient not in ref.patients.index:
                out.at[i, "matched by"] = "not in metadata"
        md["local_severity"] = pd.to_numeric(out["local_severity"], errors="coerce").where(lesional).values
        md["skin_type"] = out["skin_type"].values
        md["body_site"] = out["body_site"].fillna("").values
        gl = pd.Series(visit.reindex(pd.MultiIndex.from_arrays(
            [md["study"].astype(str), md["patient"].astype(str), tp])).values, index=md.index)
        own = pd.to_numeric(pd.Series(own_global, dtype=object).reindex(md.index), errors="coerce")
        gl = own.where(own.notna(), pd.to_numeric(gl, errors="coerce"))
        md["global_severity"] = gl.mask(md["group"].eq(E.HEALTHY), 0.0).values
        md["lesion_rank"] = own_rank.values
        md["lesion_rank"] = rank_lesions(md)
        ds.metadata_match = out
        counts = out["matched by"].value_counts()
        notes.append(f"[{name}] matched to the metadata file: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
        if ambiguous:
            notes.append(f"[{name}] {ambiguous} lesional sample(s) have no local severity: the patient has several "
                         f"lesions at that visit in the metadata file and the sample has no lesion number (code) or "
                         f"sample id that tells which one")
        if no_lesion:
            notes.append(f"[{name}] {no_lesion} second-lesion sample(s) (code X2 / L2) have no local severity: the "
                         f"metadata file has only one lesion of that patient at that visit")
        left = md["global_severity"].isna()
        if left.any():
            notes.append(f"[{name}] {int(left.sum())} sample(s) of {md.loc[left, 'patient'].nunique()} patient(s) have "
                         f"no global severity in the metadata file at their study, patient and timepoint")
    return notes


# =============================================================== matching patients across files
def _subject_keys(subject: str, study: str) -> dict:
    """Ways to read a subject id: as written, or the first / last number in it (leading zeros dropped).
    A leading study number is removed first ('1806_058' -> '058'), so it is never taken for the subject."""
    out = {"as written": subject}
    rest = re.sub(rf"^\D*{re.escape(study)}(?=\D|$)\D*", "", subject) if study else subject
    numbers = re.findall(r"\d+", rest)
    if numbers:
        out["first number"] = numbers[0].lstrip("0") or "0"
        out["last number"] = numbers[-1].lstrip("0") or "0"
    return out


def match_subjects(ds: Dataset, reference: dict) -> list:
    """Rewrite ds patient keys to the subject ids of the metadata file (reference: {study: set of ids}).
    Per study, every combination of reading rules (as written / first number / last number, on both sides) is
    tried and the one that links most patients is used, e.g. '58_AU' -> 58 = '1806_058'. Several spellings of
    one number (58_AU, 58_AX) are taken as the same person; a number that fits two subjects is left unmatched."""
    notes = []
    parts = ds.md["patient"].str.split("|", n=1, expand=True)
    new = ds.md["patient"].copy()
    rules = ("as written", "first number", "last number")
    for study in pd.unique(parts[0]):
        ref = sorted(reference.get(study, ()))
        mine = sorted(set(parts.loc[parts[0] == study, 1]))
        if not mine:
            continue
        if not ref:
            notes.append(f"[{ds.name}] study {study}: the metadata file has no patients of this study, so "
                         f"group, severity and links to other modalities are missing for {len(mine)} patient(s)")
            continue
        best = None
        for rule_ref in rules:
            lookup = {}
            for r in ref:
                k = _subject_keys(r, study).get(rule_ref)
                if k is not None:
                    lookup.setdefault(k, []).append(r)
            for rule_mine in rules:
                mapping, ambiguous = {}, []
                for m in mine:
                    k = _subject_keys(m, study).get(rule_mine)
                    hits = lookup.get(k, [])
                    if len(hits) == 1:
                        mapping[m] = hits[0]
                    elif len(hits) > 1:
                        ambiguous.append(m)
                score = (len(mapping), rule_mine == "as written", rule_ref == "as written")
                if best is None or score > best[0]:
                    best = (score, rule_mine, rule_ref, mapping, ambiguous)
        _, rule_mine, rule_ref, mapping, ambiguous = best
        sel = parts[0] == study
        new[sel] = study + "|" + parts.loc[sel, 1].map(lambda m: mapping.get(m, m))
        missing = [m for m in mine if m not in mapping]
        if mapping:
            ex = next(iter(mapping.items()))
            how = "as written" if rule_mine == rule_ref == "as written" else f"the {rule_mine} of the id"
            notes.append(f"[{ds.name}] study {study}: {len(mapping)} of {len(mine)} patients linked to the metadata "
                         f"file on {how} (e.g. '{ex[0]}' = '{ex[1]}')")
        if missing:
            notes.append(f"[{ds.name}] study {study}: {len(missing)} of {len(mine)} patients not found in the "
                         f"metadata file, e.g. {missing[:6]}; subject ids there look like {ref[:6]}"
                         + (f"; {len(ambiguous)} fit more than one subject there" if ambiguous else ""))
    ds.md["patient"] = new
    return notes


def harmonise(datasets: dict, ref: MetadataReference | None = None, priority=("skin", "gut")) -> list:
    """Link every dataset's patients to the subjects of the metadata file, then take group / severity / PASI from
    that file (and from nowhere else). Without a metadata file, patients are linked to the microbiome datasets."""
    notes = []
    order = [n for n in priority if n in datasets] + [n for n in datasets if n not in priority]
    if ref is not None:
        reference, sources = ref.subjects, [n for n in priority if n in datasets]   # microbiome ids = metadata ids
    else:
        sources = [n for n in priority if n in datasets] or order[:1]
        reference = {}
        for n in sources:
            for study, subj in datasets[n].md["patient"].str.split("|", n=1, expand=True).drop_duplicates().itertuples(index=False):
                reference.setdefault(study, set()).add(subj)
    for n in order:
        if n not in sources:
            notes += match_subjects(datasets[n], reference)
    notes += apply_patient_metadata(datasets, ref)
    for n, ds in datasets.items():
        ds.metadata_ref = ref
        md = ds.md
        labelled = md["site"].isin(["lesional", "non-lesional"]).any()
        # skin / blister samples of healthy volunteers: site 'healthy' (reference for lesional and non-lesional)
        if "healthy_match" not in md:
            md["healthy_match"] = ""
        sites = ("lesional", "non-lesional") + (tuple(ds.healthy_sites) if labelled else ())
        site, match = E.to_healthy_site(md["site"], md["group"], sites)
        moved = site.ne(md["site"].astype(object))
        md["healthy_match"] = md["healthy_match"].astype(object).fillna("").where(~moved, match)
        md["site"] = site
        not_lesional = md["site"].ne("lesional")
        md["local_severity"] = pd.to_numeric(md["local_severity"], errors="coerce").mask(not_lesional)
        md["lesion_rank"] = pd.to_numeric(md["lesion_rank"], errors="coerce").mask(not_lesional)
        for site, target in (ds.patient_site.items() if labelled else []):
            move = md["site"].eq(site) & ~md["group"].eq(E.HEALTHY)   # e.g. a patient's blister without a label
            if move.any():
                md.loc[move, "site"] = target
                notes.append(f"[{n}] {int(move.sum())} {ds.site_title(site)} sample(s) of "
                             f"{md.loc[move, 'patient'].nunique()} patient(s) have no lesional / non-lesional label; "
                             f"they are kept as a separate site '{ds.site_title(target)}'")
        ds.md = md[MD_COLS]
    return notes


# =============================================================== timepoints
def _baseline_like(label) -> bool:
    return label in E.TIMEPOINT_FIRST or bool(re.fullmatch(r"(v|t|d|day|w|wk|week|visit|occ|m|month)?[\s_.]*[01]", label))


def _guess(labels) -> dict:
    """Raw labels of one study -> canonical visit: earliest = baseline, latest = follow-up, between = mid.
    Blank and 'not applicable' labels are baseline; if they occur next to labels that do not start at a
    baseline-like name (0, 1, v1, pre ...), those other labels are all taken as later visits.
    Negative day labels (e.g. '-14') are pre-treatment screening -> baseline."""
    real = [l for l in labels if l != NO_LABEL and l not in NOT_APPLICABLE]
    ordered = E.timepoint_levels(pd.DataFrame({"timepoint": real})) if real else []
    out = {l: "baseline" for l in labels}
    later_only = len(real) < len(labels) and ordered and not _baseline_like(ordered[0])
    for i, l in enumerate(ordered):
        last = i == len(ordered) - 1
        out[l] = ("follow-up" if last else "mid") if later_only else ("baseline" if i == 0 else "follow-up" if last else "mid")
    for l in labels:
        if str(l).strip().startswith("-"):
            out[l] = "baseline"
    return out


SAME_FILE = ("skin", "gut")        # datasets whose timepoint labels come from the same metadata file


def timepoint_map(datasets: dict, saved: pd.DataFrame | None = None, ref: MetadataReference | None = None
                  ) -> pd.DataFrame:
    """Proposed mapping (dataset, study, raw label) -> canonical timepoint, one row per label in use.
    Skin and gut labels come from one metadata file and share a mapping per study; every other table is read
    on its own, because its labels (0/1/2, 1/3, ...) need not mean the same as another file's.
    `saved` rows override the proposal. Labels of the metadata file that no microbiome sample uses get rows of
    their own ('Metadata file'): they set the timepoint of the metadata rows other datasets are matched to."""
    counts = {}
    labels_of = {n: ds.label for n, ds in datasets.items()}
    for n, ds in datasets.items():
        raw = ds.raw_timepoint.astype(object).where(ds.raw_timepoint.notna(), NO_LABEL)
        for (study, label), k in ds.md.groupby(["study", raw.values]).size().items():
            counts[(n, study, str(label))] = int(k)
    if ref is not None:
        labels_of["_metadata"] = ref.label
        raw = ref.rows["raw_timepoint"].astype(object).where(ref.rows["raw_timepoint"].notna(), NO_LABEL)
        for (study, label), k in ref.rows.groupby(["study", raw.values]).size().items():
            if not any((n, study, str(label)) in counts for n in SAME_FILE):
                counts[("_metadata", study, str(label))] = int(k)
    same_file = SAME_FILE + ("_metadata",)
    rows = []
    for study in sorted({k[1] for k in counts}):
        sets = {n: {k[2] for k in counts if k[0] == n and k[1] == study} for n in labels_of}
        shared = _guess(sorted(set().union(*[sets[n] for n in same_file if n in sets])))
        for n, labels in sets.items():
            guess = shared if n in same_file else _guess(sorted(labels))
            for label in sorted(labels, key=lambda l: (l != NO_LABEL, E._natural_key(l))):
                tp = _override_timepoint(study, label) or guess.get(label, "baseline")
                rows.append({"dataset": labels_of[n], "study": study, "label in file": label,
                             "samples": counts[(n, study, label)], "timepoint": tp})
    tab = pd.DataFrame(rows)
    if saved is not None and len(saved) and len(tab):
        key = ["dataset", "study", "label in file"]
        old = saved.astype({c: str for c in key}).drop_duplicates(key).set_index(key)["timepoint"]
        hit = pd.MultiIndex.from_frame(tab[key].astype(str)).map(lambda k: old.get(k, None))
        tab["timepoint"] = [h if isinstance(h, str) and h in CANONICAL_TIMEPOINTS else t
                            for h, t in zip(hit, tab["timepoint"])]
    return tab


def _visit_key(md: pd.DataFrame) -> pd.MultiIndex:
    """study + patient + timepoint: the only key on which a score is shared between samples."""
    return pd.MultiIndex.from_arrays([md["study"].astype(str), md["patient"].astype(str),
                                      md["timepoint"].astype(object).where(md["timepoint"].notna(), "").astype(str)])


def apply_timepoint_map(datasets: dict, tab: pd.DataFrame, ref: MetadataReference | None = None) -> list:
    """Set md['timepoint'] of every dataset from the mapping table, then the visit and sample metadata from the
    metadata file (ref; default: the one harmonise used). Returns notes."""
    if ref is None:
        ref = next((r for r in (getattr(ds, "metadata_ref", None) for ds in datasets.values()) if r is not None), None)
    notes = []
    lookup = {(r["dataset"], str(r["study"]), str(r["label in file"])): r["timepoint"] for r in tab.to_dict("records")}
    for ds in datasets.values():
        raw = ds.raw_timepoint.astype(object).where(ds.raw_timepoint.notna(), NO_LABEL).astype(str)
        tp = [lookup.get((ds.label, s, l), "baseline") for s, l in zip(ds.md["study"].astype(str), raw)]
        ds.md["timepoint"] = pd.array(tp, dtype="string")
        one = ds.md[~(pd.to_numeric(ds.md["lesion_rank"], errors="coerce") > 1)]     # a second lesion is expected
        twice = one.groupby(["patient", "site", "timepoint"]).size()
        if (twice > 1).any():
            notes.append(f"[{ds.name}] {int((twice > 1).sum())} patient / site / timepoint combination(s) have more "
                         f"than one sample; they are averaged. Check the timepoint mapping.")
    for ds in datasets.values():                           # healthy volunteers sampled once (see the engine)
        ds.md["healthy_reference"] = E.timeless_healthy(ds.md.drop(columns="healthy_reference", errors="ignore"))
        n = int(ds.md.loc[ds.md["healthy_reference"], "patient"].nunique())
        several = int(ds.md.loc[ds.md["group"].eq(E.HEALTHY) & ~ds.md["healthy_reference"], "patient"].nunique())
        if n and ds.md.loc[~ds.md["healthy_reference"], "timepoint"].nunique() > 1:
            notes.append(f"[{ds.name}] {n} healthy volunteer(s) were sampled at one timepoint only: they are the "
                         f"reference at every timepoint"
                         + (f"; {several} with several timepoints keep their own timepoints" if several else ""))
    return notes + apply_sample_metadata(datasets, ref, tab)


# =============================================================== overview tables
def coverage(datasets: dict) -> pd.DataFrame:
    """Patients per study and timepoint for every modality and site."""
    cols = {}
    for ds in datasets.values():
        for site in ds.sites:
            m = ds.md[ds.md["site"] == site]
            name = ds.label if len(ds.sites) == 1 else f"{ds.label} · {ds.site_title(site)}"
            cols[name] = m.groupby(["study", m["timepoint"].astype(object)])["patient"].nunique()
    tab = pd.DataFrame(cols).fillna(0).astype(int)
    tab.index.names = ["study", "timepoint"]
    order = {t: i for i, t in enumerate(CANONICAL_TIMEPOINTS)}
    tab = tab.reset_index()
    return tab.sort_values(["study", "timepoint"], key=lambda s: s.map(order) if s.name == "timepoint" else s).reset_index(drop=True)
