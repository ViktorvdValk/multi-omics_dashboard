"""
discovery.py -- find the study data on the data drive (V2 automatic loading).

Layout (the drive is Z:\\ by default; set MULTIOMICS_DRIVE or change it on Data setup):

    Z:\\CHDR1806_SKINERGY_PSORIASIS\\          study folder: CHDR + study number + anything
        metadata\\  <metadata file(s)>          sample metadata of that study
        processed\\ <modality folder>\\ <data files, also in subfolders>

Every data file below processed\\ is listed on Data setup with the modality recognised from its CONTENT
(file names and formats can differ):
    microbiome  a Biotaviz table ('#class' / 'class id' tree, read counts per sample) or a .biom file;
                skin or gut is taken from the metadata of every sample, the file / folder name ('gut', 'faeces',
                'skin', 'swab' ...) is only a hint that is checked against it
    energy      a table with succinate / fumarate / malate / isocitrate columns (blister fluid or plasma: from the
                file or folder name, else from a Site column, else you choose)
    ceramides   a long table with a ceramide column, or a wide one with ceramide names as columns
Nothing is loaded until you press Load (one file, or Load all).

All metadata files of all study folders are combined (combine_metadata): rows are matched on study + sample_id
(rows without sample_id on study + subject + site + lesion status + timepoint), a value missing in one file is
taken from another, and where files disagree the study's own file (the one in that study's CHDR folder) wins.
Every disagreement is listed.
"""
from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import datasets as DS
import load_studies as L

DEFAULT_DRIVE = os.environ.get("MULTIOMICS_DRIVE", "Z:\\")
STUDY_FOLDER = re.compile(r"^CHDR[\s_\-]?(\d{4})", re.I)
DATA_EXT = (".xlsx", ".xlsm", ".xls", ".csv", ".tsv", ".txt", ".biom")
MAX_DEPTH = 4                     # folders below a modality folder that are searched
MODALITIES = {"microbiome": "Microbiome", "ceramides": "Ceramides", "energy": "Energy metabolism",
              "unknown": "skip (not recognised)"}
PARTS = {"microbiome": ["from metadata", "skin", "gut"], "energy": ["blister", "plasma"], "ceramides": ["-"],
         "unknown": ["-"]}
GUT_WORDS = ("gut", "faec", "fec", "feces", "faeces", "stool", "fc")
SKIN_WORDS = ("skin", "swab", "nostril", "nasal", "nose", "lesion", "sk")
METABOLITES = ("succinate", "fumarate", "malate", "isocitrate", "citrate")

# canonical metadata columns (as in meta_data_all_complete.csv) and other spellings found in metadata files
META_COLUMNS = ["sample_id", "study", "subject", "group", "site", "lesion_status", "region_location", "region",
                "global_severity", "stage", "timepoint", "local_severity"]
META_ALIASES = {"sample_id": ["sample_id", "sampleid", "sample", "sample_name", "sample_nr", "lcab_id"],
                "sample_source": ["sample_source", "sample_type", "specimen", "specimen_type"],
                "subject": ["subject", "subject_id", "subj_id", "subjid", "patient", "patient_id"],
                "timepoint": ["timepoint", "time_point", "visit", "sampling_time"],
                "lesion_status": ["lesion_status", "lesion", "location"],
                "global_severity": ["global_severity", "pasi_score", "mswat"],
                "local_severity": ["local_severity", "lesion_severity"]}


@dataclass
class Study:
    study: str
    path: Path
    metadata: list = field(default_factory=list)       # metadata files
    items: list = field(default_factory=list)          # data files (dicts, see scan)
    notes: list = field(default_factory=list)


# =============================================================== scanning
def _sub(folder: Path, name: str):
    try:
        return next((p for p in folder.iterdir() if p.is_dir() and p.name.lower() == name), None)
    except OSError:
        return None


def _files(folder: Path, depth=MAX_DEPTH):
    out = []
    try:
        entries = sorted(folder.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return out
    for p in entries:
        if p.name.startswith(("~$", ".")):
            continue
        if p.is_file() and p.suffix.lower() in DATA_EXT:
            out.append(p)
        elif p.is_dir() and depth > 0:
            out += _files(p, depth - 1)
    return out


def scan(drive: str) -> tuple[list, list]:
    """Study folders on the drive with their metadata and data files. Returns (studies, notes)."""
    root = Path(str(drive).strip().strip('"') or DEFAULT_DRIVE)
    notes = []
    if not root.is_dir():
        return [], [f"data drive / folder not found: {root}"]
    studies = []
    for folder in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
        m = STUDY_FOLDER.match(folder.name)
        if not m:
            continue
        st = Study(m.group(1), folder)
        meta = _sub(folder, "metadata")
        proc = _sub(folder, "processed")
        if meta is None:
            st.notes.append(f"{folder.name}: no 'metadata' folder")
        else:
            st.metadata = [p for p in _files(meta) if p.suffix.lower() != ".biom"]
        if proc is None:
            st.notes.append(f"{folder.name}: no 'processed' folder")
        else:
            for f in _files(proc):
                rel = f.relative_to(proc)
                modality_folder = rel.parts[0] if len(rel.parts) > 1 else ""
                st.items.append(classify(f, st.study, modality_folder, folder.name))
        studies.append(st)
    if not studies:
        notes.append(f"no study folders (CHDR<number>...) in {root}")
    return studies, notes


# =============================================================== recognising a file
def read_head(path: Path, nrows=40) -> dict:
    """First rows of every sheet (Excel) or of the text, without header, as text: {sheet: DataFrame}."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm"):
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        out = {}
        for ws in wb.worksheets:
            rows = []
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                if i >= nrows:
                    break
                rows.append(["" if v is None else str(v) for v in row])
            if rows:
                width = max(len(r) for r in rows)
                out[ws.title] = pd.DataFrame([r + [""] * (width - len(r)) for r in rows])
        wb.close()
        return out
    if suffix == ".xls":
        return {k: v.astype(str) for k, v in pd.read_excel(path, sheet_name=None, header=None, nrows=nrows,
                                                            dtype=str).items()}
    with open(path, "rb") as fh:
        text = fh.read(200_000).decode("utf-8-sig", errors="replace")
    lines = text.splitlines()[:nrows]
    sep = _sep(lines[0] if lines else "")
    return {"": pd.read_csv(io.StringIO("\n".join(lines)), sep=sep, header=None, dtype=str,
                            on_bad_lines="skip", engine="python").fillna("")}


def _sep(header: str) -> str:
    return "\t" if "\t" in header else ";" if ";" in header else ","


def _words(text: str) -> set:
    return set(re.findall(r"[a-z]+", str(text).lower()))


def hint(text: str, words) -> bool:
    w = _words(text)
    return any(x in w or any(t.startswith(x) for t in w if len(x) > 2) for x in words)


def classify(path: Path, study: str, modality_folder: str, study_folder: str) -> dict:
    """What a data file holds, from its content (and its name for skin / gut and blister / plasma)."""
    item = {"id": str(path), "path": str(path), "study": study, "study folder": study_folder,
            "modality folder": modality_folder, "file": path.name, "sheet": None, "modality": "unknown",
            "part": "-", "why": "", "size kB": round(path.stat().st_size / 1024, 1) if path.exists() else np.nan}
    name_text = f"{modality_folder} {path.stem}"
    if path.suffix.lower() == ".biom":
        item.update(modality="microbiome", why="BIOM file")
    else:
        try:
            heads = read_head(path)
        except Exception as exc:                                         # unreadable: listed, not loadable
            item["why"] = f"cannot read: {exc}"
            return item
        for sheet, head in heads.items():
            kind, why = _kind_of(head)
            if kind != "unknown":
                item.update(modality=kind, why=why, sheet=sheet or None)
                break
        else:
            item["why"] = "no Biotaviz tree, metabolite or ceramide columns found"
    if item["modality"] == "microbiome":
        gut, skin = hint(name_text, GUT_WORDS), hint(name_text, SKIN_WORDS)
        item["part"] = "gut" if gut and not skin else "skin" if skin and not gut else "from metadata"
    elif item["modality"] == "energy":
        low = name_text.lower()
        item["part"] = "plasma" if "plasma" in low or "serum" in low else "blister"
    return item


def _kind_of(head: pd.DataFrame) -> tuple[str, str]:
    if L.looks_like_biotaviz(head):
        return "microbiome", "Biotaviz tree"
    cells = {re.sub(r"[\s\-]+", "_", str(v).strip().lower()) for v in head.iloc[:5].values.ravel() if str(v).strip()}
    if any(m in cells for m in METABOLITES):
        return "energy", "metabolite columns (" + ", ".join(m for m in METABOLITES if m in cells) + ")"
    if "ceramide" in cells or "ceramides" in cells:
        return "ceramides", "ceramide column"
    from ceramide_loader import parse_ceramide
    named = [c for c in cells if parse_ceramide(c).get("cer_class", "unknown") != "unknown"]
    if len(named) >= 3:
        return "ceramides", f"{len(named)} ceramide names as columns"
    return "unknown", ""


def read_full(path: str, sheet=None) -> pd.DataFrame:
    """The whole table as text, with its header row (sheet: Excel sheet name)."""
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(p, sheet_name=sheet or 0, dtype=str)
    with open(p, "rb") as fh:
        text = fh.read().decode("utf-8-sig", errors="replace")
    return pd.read_csv(io.StringIO(text), sep=_sep(text.split("\n", 1)[0]), dtype=str)


def read_raw(path: str, sheet=None) -> pd.DataFrame:
    """The whole table as text WITHOUT header (Biotaviz)."""
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(p, sheet_name=sheet or 0, dtype=str, header=None)
    with open(p, "rb") as fh:
        text = fh.read().decode("utf-8-sig", errors="replace")
    return pd.read_csv(io.StringIO(text), sep=_sep(text.split("\n", 1)[0]), dtype=str, header=None)


# =============================================================== which metadata files belong to the loaded data
def cell_values(path: str) -> frozenset:
    """Every non-empty cell of a metadata file, lower case (to find the files that hold a data file's sample ids)."""
    try:
        df = read_full(str(path))
    except Exception:
        return frozenset()
    vals = {_clean(v).lower() for v in df.values.ravel()}
    return frozenset(v for v in vals if v)


def match_metadata(chosen: list, studies: list, ids_of, values_of) -> tuple[list, pd.DataFrame, list]:
    """The metadata files to combine for the loaded data files `chosen`: every metadata file of their own study
    folders, plus metadata files of OTHER study folders that hold sample ids of a loaded file (e.g. a shared
    sequencing run). ids_of(item) -> set of the file's sample ids (lower case); values_of(path) -> cell values.
    Returns ([(path, folder study)], table data file -> metadata files, notes)."""
    folder_of = {s.study: s for s in studies}
    all_meta = [(str(p), s.study) for s in studies for p in s.metadata]
    used, rows, notes = {}, [], []
    for it in chosen:
        own = [(p, st) for p, st in all_meta if st == it["study"]]
        ids = ids_of(it)
        hits = {}
        for p, st in all_meta:
            n = len(ids & values_of(p)) if ids else 0
            if n:
                hits[p] = n
        extra = [(p, st) for p, st in all_meta if st != it["study"] and hits.get(p)]
        for p, st in own + extra:
            used.setdefault(p, st)
        if not own and not extra:
            notes.append(f"[metadata] {it['file']}: no metadata file in {folder_of[it['study']].path.name}/metadata "
                         f"and none elsewhere holds its sample ids")
        elif ids and not any(hits.get(p) for p, _ in own + extra):
            notes.append(f"[metadata] {it['file']}: none of its {len(ids)} sample ids (e.g. {sorted(ids)[:3]}) is in "
                         f"a metadata file; it is matched on patient + site + timepoint only")
        rows.append({"study": it["study"], "data file": it["file"],
                     "metadata files": ", ".join(f"{Path(p).name}" + (f" ({hits[p]} ids)" if hits.get(p) else "")
                                                 for p, _ in own + extra) or "none",
                     "sample ids found": f"{sum(1 for i in ids if any(i in values_of(p) for p, _ in own + extra))}"
                                         f" of {len(ids)}" if ids else "-"})
    return list(used.items()), pd.DataFrame(rows), notes


# =============================================================== combining the metadata files
def _std_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"[\s\-]+", "_", str(c).strip().lower()) for c in df.columns]
    ren = {}
    for target, aliases in META_ALIASES.items():
        if target in df.columns:
            continue
        hit = next((a for a in aliases if a in df.columns and a not in ren), None)
        if hit:
            ren[hit] = target
    df = df.rename(columns=ren)
    return df.loc[:, ~pd.Index(df.columns).duplicated()]


def _clean(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none", "<na>") else s


def _same(a: str, b: str) -> bool:
    if a == b:
        return True
    x, y = DS.to_number(pd.Series([a, b]))
    if pd.notna(x) and pd.notna(y):
        return bool(np.isclose(x, y))
    return a.strip().lower() == b.strip().lower()


def combine_metadata(files: list) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """files: [(path, study of its CHDR folder)]. Returns (combined metadata, discrepancies, notes).
    One row per study + sample_id (rows without sample_id: study + subject + site + lesion status + timepoint).
    A value missing in one file is taken from another; where files disagree the file of the study's own folder
    wins (if its files agree among themselves), else the value stays missing. Every disagreement is listed."""
    notes, frames = [], []
    for path, folder_study in files:
        try:
            raw = read_full(str(path))
        except Exception as exc:
            notes.append(f"[metadata] {Path(path).name}: cannot read ({exc})")
            continue
        df = _std_columns(raw).astype(object)
        if "study" not in df:
            df["study"] = folder_study
        df["study"] = df["study"].map(_clean).map(lambda v: DS.norm_study(v) if v else folder_study)
        df["_file"] = Path(path).name
        df["_own"] = df["study"].eq(folder_study)
        df["_order"] = len(frames)
        frames.append(df)
        notes.append(f"[metadata] {Path(path).parent.parent.name}/{Path(path).name}: {len(df)} rows "
                     f"(studies {', '.join(sorted(df['study'].unique()))})")
    if not frames:
        raise FileNotFoundError("no readable metadata file")
    allrows = pd.concat(frames, ignore_index=True, sort=False)
    for c in META_COLUMNS:
        if c not in allrows:
            allrows[c] = ""
    data_cols = [c for c in allrows.columns if not c.startswith("_")]
    for c in data_cols:
        allrows[c] = allrows[c].map(_clean)
    flags = _unify_subjects(allrows, notes)
    sid = allrows["sample_id"]
    alt = ("row:" + allrows["subject"] + "|" + allrows["site"].str.lower() + "|" + allrows["lesion_status"].str.lower()
           + "|" + allrows["timepoint"])
    allrows["_key"] = allrows["study"] + "|" + sid.where(sid.ne(""), alt)

    merged = []
    for key, g in allrows.groupby("_key", sort=False):
        row = {}
        for c in data_cols:
            vals = [(v, f, o) for v, f, o in zip(g[c], g["_file"], g["_own"]) if v != ""]
            if not vals:
                row[c] = ""
                continue
            distinct = []
            for v, _, _ in vals:
                if not any(_same(v, d) for d in distinct):
                    distinct.append(v)
            if len(distinct) == 1:
                row[c] = distinct[0]
                continue
            own = [v for v, _, o in vals if o]
            own_distinct = []
            for v in own:
                if not any(_same(v, d) for d in own_distinct):
                    own_distinct.append(v)
            if len(own_distinct) == 1:
                used, rule = own_distinct[0], "the study's own file"
            else:
                used, rule = "", ("the study's own file disagrees with itself: left missing" if own_distinct
                                  else "no file of the study's own folder has it: left missing")
            row[c] = used
            flags.append({"kind": "same sample in several rows" if g["_file"].nunique() == 1 else "between files",
                          "study": g["study"].iloc[0], "item": key.split("|", 1)[1], "field": c, "used": used,
                          "values": "; ".join(f"{v} ({f})" for v, f, _ in vals), "files": ", ".join(g["_file"].unique()),
                          "rule": rule})
        row["_files"] = ", ".join(g["_file"].unique())
        merged.append(row)
    combined = pd.DataFrame(merged)
    combined = combined[META_COLUMNS + [c for c in combined.columns if c not in META_COLUMNS]]
    flags += _patient_level_flags(combined)
    disc = pd.DataFrame(flags, columns=["kind", "study", "item", "field", "used", "values", "files", "rule"])
    n_multi = int((allrows.groupby("_key")["_file"].nunique() > 1).sum())
    notes.append(f"[metadata] combined {len(allrows)} rows of {len(frames)} file(s) into {len(combined)} samples"
                 + (f"; {n_multi} sample(s) are in more than one file" if n_multi else "")
                 + (f"; {len(disc)} discrepancies (see the list on Data setup)" if len(disc) else "; no discrepancies"))
    return combined, disc, notes


def _unify_subjects(allrows: pd.DataFrame, notes: list) -> list:
    """One spelling per patient across the metadata files of a study. The patients of the rows that have a group
    (the clinical metadata) are the reference; a subject of another file that is none of them is linked by its
    number when that fits exactly one reference patient (e.g. '200_AN' in the energy metadata = '200'), as the
    data files are linked (datasets.match_subjects). Changes allrows['subject'] in place; returns flags."""
    flags = []
    for study, g in allrows.groupby("study"):
        subj = g["subject"].map(lambda v: DS.norm_subject(v) if v else "")
        ref = set(subj[g["group"].ne("") & subj.ne("")])
        if not ref:
            continue
        keys = {r: DS._subject_keys(r, study) for r in ref}
        lookup = {}
        for rule in ("as written", "first number", "last number"):
            for r, k in keys.items():
                if rule in k:
                    lookup.setdefault((rule, k[rule]), set()).add(r)
        linked = {}
        for other in sorted(set(subj[subj.ne("")]) - ref):
            mine = DS._subject_keys(other, study)
            for rule_mine, rule_ref in (("as written", "first number"), ("first number", "as written"),
                                        ("first number", "first number"), ("last number", "last number")):
                hit = lookup.get((rule_ref, mine.get(rule_mine)), set())
                if len(hit) == 1:
                    linked[other] = next(iter(hit))
                    break
        if not linked:
            continue
        change = subj.isin(list(linked))
        for other, target in linked.items():
            files = ", ".join(sorted(set(g.loc[subj.eq(other), "_file"])))
            flags.append({"kind": "patient id", "study": study, "item": target, "field": "subject", "used": target,
                          "values": f"{other} ({files}); {target}", "files": files,
                          "rule": "the same patient number: the spelling of the clinical metadata is used"})
        allrows.loc[change[change].index, "subject"] = subj[change].map(linked)
        ex = next(iter(linked.items()))
        notes.append(f"[metadata] study {study}: {len(linked)} patient id(s) of other metadata files linked to the "
                     f"clinical metadata by number (e.g. '{ex[0]}' = '{ex[1]}')")
    return flags


def _patient_level_flags(m: pd.DataFrame) -> list:
    """Values that should be one per patient (group, stage) or per visit (global severity) but differ between
    rows of the combined metadata."""
    out = []
    for field_, keys, rule in (
            ("group", ["study", "subject"], "one group per patient: skin rows first, 'healthy' wins"),
            ("stage", ["study", "subject"], "one stage per patient: the first in the file is used"),
            ("global_severity", ["study", "subject", "timepoint"],
             "per visit: every sample keeps its own row's value; samples without one stay missing")):
        sub = m[m[field_].ne("") & m["subject"].ne("")]
        for k, g in sub.groupby(keys):
            distinct = []
            for v in g[field_]:
                if not any(_same(v, d) for d in distinct):
                    distinct.append(v)
            if len(distinct) > 1:
                vals = g.groupby(field_)["sample_id"].apply(lambda s: ", ".join(s.astype(str)[:4]))
                out.append({"kind": "within patient" if field_ != "global_severity" else "within visit",
                            "study": k[0], "item": " | ".join(map(str, k[1:])), "field": field_, "used": "",
                            "values": "; ".join(f"{v} (samples {s})" for v, s in vals.items()),
                            "files": ", ".join(sorted(set(", ".join(g["_files"]).split(", ")))), "rule": rule})
    return out
