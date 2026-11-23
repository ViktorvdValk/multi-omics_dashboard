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


def energy_dataset(tables: dict) -> DS.Dataset:
    """tables = {'blister': DataFrame, 'plasma': DataFrame}; a key is the specimen type of that file. A value can
    also be a list of (DataFrame, study of its folder, file name) to combine several files of one specimen."""
    parts, notes = [], []
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
