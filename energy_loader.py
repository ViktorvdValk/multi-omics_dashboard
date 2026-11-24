"""
energy_loader.py -- energy metabolism (mass spectrometry): succinate, fumarate, malate and isocitrate in
blister fluid and in plasma, one file per specimen type, study 1806.

File layout (tab-separated with decimal commas, or Excel):
    LCAB_ID          Subj_ID  Sampling_time  Site  Succinate  Fumarate  Malate  Isocitrate
    TES0019_1001_A   53_AN    0              L     1,100      0,248     1,331   178,199

Subj_ID is matched to the subject of the metadata file (see datasets.match_subjects); Sampling_time is
the file's own timepoint label ('nvt' = not applicable, used for healthy volunteers with a single sample).
Every metabolite is analysed on its own ln scale; nothing is summed across metabolites.

Blister fluid is sampled at a lesional (perilesional) and a non-lesional site of every patient; the blister
file's Site column (L / NL, lesional / perilesional / non-lesional ...) says which: the sites 'lesional' and
'non-lesional', with the same paired analyses as the skin samples. Healthy volunteers have one blister without a
site label: the site 'healthy', the healthy reference for both the lesional and the non-lesional blisters
(skin_engine.SHARED_HEALTHY_SITES). Plasma is one site, 'plasma', whatever the plasma file says.
"""
from __future__ import annotations

import re

import pandas as pd

import datasets as DS

FEATURE_ALIASES = ("metabolite", "analyte", "parameter", "test", "compound", "marker", "feature", "variable")
DEFAULT_STUDY = "1806"          # used when a file has no study column
SKIN_SITES = ("lesional", "non-lesional")
SITE_TITLES = {"blister-unlabelled": "blister (patient, no site label)"}


def _has_site_column(raw: pd.DataFrame) -> bool:
    return any(re.sub(r"[\s\-]+", "_", str(c).strip().lower()) in DS.COLUMN_ALIASES["site"] for c in raw.columns)


def _source_label(row: dict) -> str:
    """The specimen / site label of a metadata row: its 'sample source' (e.g. 'blister perilesional',
    'blister non-lesional', 'blister', 'plasma'), else its site when that names a blister or plasma."""
    src = str(row.get("sample_source") or "").strip()
    if not src:
        site = str(row.get("site") or "").strip()
        src = site if re.search(r"blister|plasma|serum", site, re.I) else ""
    return src


def split_by_metadata(raw: pd.DataFrame, specimen: str, study: str, meta: pd.DataFrame | None):
    """Site (and blister / plasma) of every row of an energy file from its metadata row, matched by the file's
    sample id (LCAB_ID / sample_id ...) = the metadata sample_id. The metadata column 'sample source' says
    'blister perilesional' (lesional), 'blister non-lesional', 'blister' (healthy volunteer) or 'plasma'.
    A Site column in the file itself is kept. Rows without a metadata match keep `specimen` (the part chosen
    on Data setup). Returns ({specimen: DataFrame}, notes)."""
    if meta is None or meta.empty:
        return {specimen: raw}, []
    norm = [re.sub(r"[\s\-]+", "_", str(c).strip().lower()) for c in raw.columns]
    id_col = next((raw.columns[norm.index(c)] for c in DS.ID_COLUMNS if c in norm), None)
    if id_col is None:
        return {specimen: raw}, [f"[energy] {specimen} file without a sample id column (LCAB_ID / sample_id): "
                                 f"the site cannot be taken from the metadata 'sample source'"]
    rows = meta.assign(_sid=meta["sample_id"].fillna("").astype(str).str.strip().str.lower(),
                       _study=meta["study"].fillna("").astype(str).str.strip())
    rows = rows[rows["_sid"].ne("")]
    own = {r["_sid"]: _source_label(r) for r in rows[rows["_study"] == str(study)].to_dict("records")}
    other = {r["_sid"]: _source_label(r) for r in rows.to_dict("records")}
    sid = raw[id_col].fillna("").astype(str).str.strip().str.lower()
    src = sid.map(lambda v: own.get(v) or other.get(v) or "")
    kind = src.str.lower().map(lambda v: "plasma" if ("plasma" in v or "serum" in v)
                               else ("blister" if "blister" in v else ""))
    kind = kind.where(kind.ne(""), specimen)
    has_site = _has_site_column(raw)
    out, notes = {}, []
    for k, idx in kind.groupby(kind).groups.items():
        part = raw.loc[idx].copy()
        if k == "blister" and not has_site:
            part["Site"] = src.loc[idx].where(src.loc[idx].ne(""), None).values
        out[k] = part
    found = int(src.ne("").sum())
    notes.append(f"[energy] {specimen} file: {found} of {len(raw)} rows have a 'sample source' in the metadata "
                 f"(by {id_col}); " + ", ".join(f"{k}: {len(v)} rows" for k, v in out.items())
                 + ("; the file's own Site column is kept" if has_site else ""))
    if found < len(raw):
        notes.append(f"[energy] {len(raw) - found} row(s) without a metadata 'sample source' (e.g. "
                     f"{raw.loc[src.eq(''), id_col].astype(str).tolist()[:4]}) are kept as {specimen}"
                     + ("" if specimen != "blister" or has_site else " without a site label"))
    return out, notes


def energy_dataset(tables: dict, notes_in=()) -> DS.Dataset:
    """tables = {'blister': DataFrame, 'plasma': DataFrame}; a key is the specimen type of that file. A value can
    also be a list of (DataFrame, study of its folder, file name) to combine several files of one specimen."""
    parts, notes = [], list(notes_in)
    flat = []
    for specimen, value in tables.items():
        for df, study, _ in (value if isinstance(value, list) else [(value, None, "")]):
            flat.append((specimen, df, study))
    for specimen, raw, study in flat:
        tidy, n = DS.tidy_table(raw, FEATURE_ALIASES, "energy", default_study=study or DEFAULT_STUDY,
                                default_site=specimen)
        notes += n
        if specimen == "blister":
            if not _has_site_column(raw):
                notes.append("[energy] the blister file has no Site column: lesional and non-lesional blisters "
                             "cannot be told apart and are kept as one site 'blister'")
            else:
                odd = sorted(set(tidy["site"]) - set(SKIN_SITES) - {"blister", "healthy"})
                if odd:
                    notes.append(f"[energy] blister Site value(s) {odd} are not lesional / non-lesional; they are "
                                 f"kept as separate site(s)")
                per = tidy.drop_duplicates(["subject", "timepoint", "site"])["site"].value_counts()
                notes.append("[energy] blister samples per Site label: "
                             + ", ".join(f"{'no label' if k == 'blister' else k} {v}" for k, v in per.items()))
        else:                                   # plasma: one site
            tidy["site"] = specimen
            tidy["_second_lesion"] = False
        parts.append(tidy)
    ds = DS.abundance_dataset("energy", "Energy metabolism", pd.concat(parts, ignore_index=True), notes, "metabolite")
    ds.site_titles = dict(SITE_TITLES)
    ds.healthy_sites = ("blister",)                       # a healthy volunteer's unlabelled blister -> 'healthy'
    ds.patient_site = {"blister": "blister-unlabelled"}   # a patient's unlabelled blister stays apart
    return ds


def refresh_ratios(ds: DS.Dataset):
    """(Re)build the ratio indices from the current values: ln(numerator / denominator) per sample."""
    cols = {f"ln({a} / {b})": ds.values[a] - ds.values[b] for a, b in ds.ratio_defs
            if a in ds.values and b in ds.values}
    ds.derived = pd.DataFrame(cols, index=ds.values.index) if cols else None


def add_ratio(ds: DS.Dataset, numerator: str, denominator: str) -> str:
    if (numerator, denominator) not in ds.ratio_defs:
        ds.ratio_defs.append((numerator, denominator))
    refresh_ratios(ds)
    return f"ln({numerator} / {denominator})"
