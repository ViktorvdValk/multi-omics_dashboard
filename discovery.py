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

import collections
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


# =============================================================== NGID (multi-sheet) metadata workbooks
# The microbiome metadata files are NGID workbooks with one sheet per entity
# (Investigation, Study, ObservationUnit, 'Sample - NGID', Assay). The sample rows of interest
# are on the 'Sample - NGID' sheet; reading sheet 0 (Investigation) gives junk rows.
# Their sample IDs are not the sequencing (DS##NBD##) IDs of the Biotaviz files:
#   1806  CMD00304_DS01NBD94      -> strip the prefix        -> DS01NBD94
#   1947  1394_DS01NBD95          -> strip the prefix        -> DS01NBD95
#   1947  1393_25443B0F7 (hex)    -> hex barcode; the DS id  -> from the mapping tables
#   2230  1393_25632A1B1 (hex)    -> same
# The hex -> DS mapping tables (meta_table1947.csv / meta_table_2230.csv /
# CHDR1806_16S_update.mapping.xlsx) live next to the study folders; the patient, site, lesion,
# visit and batch of the sheet are used to sanity-check / infer the match.
_HEX_RE = re.compile(r"^[A-Fa-f0-9]{6,16}$")
_DS_RE = re.compile(r"^DS\d+NBD\d+$", re.I)


def _hex_maps(start_path) -> dict:
    """hex swab barcode -> DS##NBD## id, from the bridge file shipped with the dashboard
    (sample_id_bridge.csv, next to this file). It is built from the study's meta_tables at
    prototyping time but, once in place, is all the loader needs — so the external meta_table*.csv
    and the 1806 mapping workbook do NOT have to be present in deployment. Falls back to those
    tables (up the tree) only if the bridge file is absent."""
    out = {}
    here = Path(__file__).resolve().parent
    bridge = here / "sample_id_bridge.csv"
    if bridge.is_file():
        try:
            t = pd.read_csv(bridge, dtype=str, comment="#")
            cols = {re.sub(r"[\s\-]+", "_", str(c).strip().lower()): c for c in t.columns}
            hcol = cols.get("sample_identifier") or cols.get("hex") or cols.get("barcode")
            scol = cols.get("sequencing_id") or cols.get("ds") or cols.get("seqid")
            if hcol and scol:
                for h, s in zip(t[hcol], t[scol]):
                    if pd.notna(h) and pd.notna(s):
                        out.setdefault(str(h).strip().split("_")[-1].upper(), str(s).strip().upper())
        except Exception:
            pass
    if out:
        return out
    # fallback: the prototyping meta_tables (not present in deployment)
    dirs = []
    p = Path(start_path)
    if p.is_file():
        p = p.parent
    cur = p
    for _ in range(8):
        dirs.append(cur)
        if cur == cur.parent:
            break
        cur = cur.parent
    for base in dirs:
        try:
            if not base.is_dir():
                continue
        except OSError:
            continue
        for name in ("meta_table1947.csv", "meta_table_2230.csv"):
            f = base / name
            if f.is_file():
                try:
                    t = pd.read_csv(f, dtype=str)
                    hcol = next((c for c in t.columns if re.sub(r"[\s\-]", "_", str(c).strip().lower()) == "samplename"), None)
                    scol = next((c for c in t.columns if re.sub(r"[\s\-]", "_", str(c).strip().lower()) == "seqid"), None)
                    if hcol and scol:
                        for h, s in zip(t[hcol], t[scol]):
                            if pd.notna(h) and pd.notna(s):
                                out.setdefault(str(h).strip().upper(), str(s).strip().upper())
                except Exception:
                    pass
        for name in ("CHDR1806_16S_update.mapping.xlsx",):
            f = base / name
            if f.is_file():
                try:
                    t = pd.read_excel(f, dtype=str)
                    hcol = scol = None
                    for c in t.columns:
                        if re.sub(r"[\s\-]+", "_", str(c).strip().lower()) == "sample-id":
                            hcol = c
                        if re.sub(r"[\s\-]+", "_", str(c).strip().lower()) == "cm_id":
                            scol = c
                    if hcol is not None and scol is not None:
                        for h, s in zip(t[hcol], t[scol]):
                            if pd.notna(h) and pd.notna(s):
                                out.setdefault(str(h).strip().upper(), str(s).strip().upper())
                except Exception:
                    pass
    return out


def _bridged_id(raw: str, study: str, hexmap: dict) -> tuple[str, str]:
    """(canonical DS id, how it was bridged)."""
    s = str(raw).strip()
    if _DS_RE.match(s):
        return s.upper(), "as written"
    # prefixed: CMD00304_DS01NBD94 / 1394_DS01NBD95
    m = re.match(r"^[A-Za-z0-9]+_+(DS\d+NBD\d+)$", s, re.I)
    if m:
        return m.group(1).upper(), "prefix stripped"
    # hex barcode: 1393_25632A1B1 / 1394_43969CCDD / bare hex
    parts = s.split("_")
    for cand in (parts[-1], s):
        if _HEX_RE.match(cand.strip()) and cand.strip().upper() in hexmap:
            return hexmap[cand.strip().upper()], "hex barcode -> mapping table"
    return s.upper(), "unrecognised"


_LESION_STATUS = {"l1": "lesional", "l2": "lesional", "l3": "lesional", "lesional": "lesional",
                  "lesion": "lesional", "nl": "non-lesional", "non-lesional": "non-lesional",
                  "nonlesional": "non-lesional", "rnos": "nostril", "rnose": "nostril",
                  "nasal": "nostril", "nose": "nostril", "healthy": "healthy", "hv": "healthy"}
_MF_STAGES = {"ia", "ib", "iia", "iib", "iiia", "iiia1", "iiia2", "iva", "iva1", "iva2",
              "fmf", "sezary", "mf"}


def _derive_group_stage(study: str, diagnosis: str, lesion_type: str) -> tuple[str, str]:
    """(group, stage) of a sample from the NGID sheet's diagnosis / lesion_type.
    1806 = psoriasis / healthy (no MF stage); 1947 / 2230 = MF (stage = the MF diagnosis) or healthy."""
    diag = diagnosis.strip().lower()
    lt = lesion_type.strip().lower()
    if lt in ("healthy", "hv") or diag in ("healthy", "hv"):
        return "healthy", ""
    if diag == "psoriasis":
        return "psoriasis", ""
    if diag in _MF_STAGES or lt in ("mf",):
        stage = diagnosis.strip() if diag in _MF_STAGES else ""
        return "MF", stage
    return "", ""


def read_ngid_metadata(path: str, study: str, hexmap: dict) -> tuple[pd.DataFrame, list]:
    """The 'Sample - NGID' sheet of an NGID workbook as a flat metadata frame (one row per sample),
    with the sample IDs bridged to the DS##NBD## sequencing IDs of the Biotaviz files.
    Returns (frame, notes). Empty frame if the file is not an NGID workbook."""
    p = Path(path)
    if p.suffix.lower() not in (".xlsx", ".xlsm"):
        return pd.DataFrame(), []
    from openpyxl import load_workbook
    wb = load_workbook(p, read_only=True, data_only=True)
    sn = next((s for s in wb.sheetnames if re.match(r"(?i)^sample", s)), None)
    if sn is None:
        wb.close()
        return pd.DataFrame(), []
    ws = wb[sn]
    it = ws.iter_rows(values_only=True)
    try:
        hdr = [re.sub(r"[\s\-]+", "_", str(h).strip().lower()) if h is not None else "" for h in next(it)]
    except StopIteration:
        wb.close()
        return pd.DataFrame(), []
    col = {h: i for i, h in enumerate(hdr) if h}
    need = "sample_identifier"
    if need not in col:
        wb.close()
        return pd.DataFrame(), []
    notes = []
    rows = []
    bridged = collections.Counter()
    for r in it:
        raw = r[col[need]]
        if raw is None or not str(raw).strip():
            continue
        get = lambda name: (str(r[col[name]]).strip() if name in col and col[name] < len(r) and r[col[name]] is not None else "")
        # Prefer the workbook's own 'sequencing_id' column (the DS##NBD## id, baked in at build time):
        # it needs no external mapping table, so this works in deployment where meta_table*.csv are absent.
        seq = get("sequencing_id")
        if re.match(r"^[A-Za-z0-9]*DS\d+NBD\d+$", str(seq).strip(), re.I):
            m = re.search(r"(DS\d+NBD\d+)$", str(seq).strip(), re.I)
            ds, how = m.group(1).upper(), "sequencing_id column"
        else:
            ds, how = _bridged_id(raw, study, hexmap)
        bridged[how] += 1
        ou = get("observation_unit_identifier")
        src = get("sample_source").lower()
        # site: gut / blister / plasma first (energy), then skin (lesion words contain 'lesion' too)
        if re.search(r"faec|fec|stool", src):
            site = "gut"
        elif "blister" in src:
            site = "blister"
        elif re.search(r"plasma|serum|blood", src):
            site = "plasma"
        elif re.search(r"skin|swab|nostril|nasal|lesion", src):
            site = "skin"
        else:
            site = ""
        # patient number: the number after the study's own prefix (dual OUs like 1393_22_1394_13 carry both)
        subject = ""
        pref = {"1806": "CMD00304", "1947": "1394", "2230": "1393"}.get(study, study)
        m = re.search(re.escape(pref) + r"_(\d+)", ou)
        if m:
            subject = m.group(1)
        elif ou:
            nums = re.findall(r"\d+", ou)
            subject = nums[-1] if nums else ou
        les = get("lesion_site").lower()
        lesion_status = _LESION_STATUS.get(les, _LESION_STATUS.get(les.split(" ")[0], ""))
        diag = get("diagnosis")
        lt = get("lesion_type")
        group, stage = _derive_group_stage(study, diag, lt)
        rows.append({
            "sample_id": ds, "raw_id": str(raw).strip(), "study": study, "subject": subject,
            "site": site, "sample_source": get("sample_source"), "lesion_status": lesion_status,
            "timepoint": get("visit") or get("sampling_time") or get("occ") or get("occasion"),
            "group": group, "stage": stage, "global_severity": "", "local_severity": "",
            "diagnosis": diag, "lesion_type": lt,
            "fitzpatrick_score": get("fitzpatrick_score") or get("fitzpatrick"),
            "region_location": get("geographic_location_(country_and_or_sea)") or get("geographic_location"),
            "batch": get("sequencing_batch"), "_files": p.name, "_bridged": how,
        })
    wb.close()
    if not rows:
        return pd.DataFrame(), notes
    df = pd.DataFrame(rows)
    unrec = int(bridged.get("unrecognised", 0))
    if unrec:
        ex = df.loc[df["_bridged"] == "unrecognised", "raw_id"].tolist()[:5]
        notes.append(f"[metadata] {p.name}: {unrec} sample id(s) could not be bridged to a DS##NBD## id "
                     f"(no mapping table found, e.g. {ex}); they will only match on patient + site + timepoint")
    if bridged.get("hex barcode -> mapping table"):
        notes.append(f"[metadata] {p.name}: {bridged['hex barcode -> mapping table']} hex barcode sample id(s) "
                     f"bridged to DS##NBD## ids via the mapping tables")
    return df, notes


# =============================================================== which metadata files belong to the loaded data
def cell_values(path: str) -> frozenset:
    """Every non-empty cell of a metadata file, lower case (to find the files that hold a data file's sample ids)."""
    try:
        p = Path(path)
        if p.suffix.lower() in (".xlsx", ".xlsm"):
            from openpyxl import load_workbook
            wb = load_workbook(p, read_only=True, data_only=True)
            if any(re.match(r"(?i)^sample", s) for s in wb.sheetnames):
                m = re.search(r"CHDR\s*[_\-]?(\d{4})", p.parent.parent.name) or re.search(r"CHDR\s*[_\-]?(\d{4})", p.parent.name)
                study = m.group(1) if m else ""
                hexmap = _hex_maps(path)
                df, _ = read_ngid_metadata(path, study, hexmap)
                wb.close()
                if not df.empty:
                    vals = {_clean(v).lower() for v in df.values.ravel()}
                    return frozenset(v for v in vals if v)
        df = read_full(str(path))
    except Exception:
        return frozenset()
    vals = {_clean(v).lower() for v in df.values.ravel()}
    return frozenset(v for v in vals if v)


def _name_features(name: str) -> dict:
    """Name tokens used to match a data file to its metadata file by name: study number, location
    (skin / gut / blister / plasma), version (v1 / v2 — data V2 must pair with metadata V2), and the
    full token set (author, date — the other v1/v2 discriminators)."""
    low = str(name).lower()
    study = ""
    for n in ("1806", "1947", "2230"):
        if re.search(r"(?<!\d)" + n + r"(?!\d)", low):
            study = n
            break
    loc = ""
    if re.search(r"skin|swab|nostril|nasal|\bsk\b|lesion", low):
        loc = "skin"
    elif re.search(r"gut|faec|fec|stool|faeces", low):
        loc = "gut"
    elif "blister" in low:
        loc = "blister"
    elif re.search(r"plasma|serum", low):
        loc = "plasma"
    mv = re.findall(r"v(\d+)", low)
    version = max(int(x) for x in mv) if mv else None
    toks = {t for t in re.findall(r"[a-z0-9]{2,}", low)
            if t not in {"xlsx", "xls", "xlsm", "rel", "abs"}}
    return {"study": study, "loc": loc, "version": version, "toks": toks}


def _name_score(data_name: str, meta_name: str) -> float:
    d = _name_features(data_name)
    m = _name_features(meta_name)
    score = 0.0
    if d["study"] and d["study"] == m["study"]:
        score += 3.0
    if d["loc"] and d["loc"] == m["loc"]:
        score += 4.0
    # version: a data file labelled V2 must pair with the metadata V2 (and V1 with V1)
    if d["version"] is not None and m["version"] is not None:
        score += 2.5 if d["version"] == m["version"] else -2.5
    score += len(d["toks"] & m["toks"]) * 1.0
    return score


def match_metadata(chosen: list, studies: list, ids_of, values_of) -> tuple[list, pd.DataFrame, list]:
    """The metadata files to combine for the loaded data files `chosen`: for EACH loaded data file, the ONE
    metadata file with the most similar name (its study number + location must match; the closest name wins).
    So only the metadata table that actually describes a dataset is used — not every table in the study
    folder. ids_of(values_of) are still used to report how many of a file's sample ids each dataset holds.
    Returns ([(path, folder study)], table data file -> metadata files, notes)."""
    all_meta = [(str(p), s.study) for s in studies for p in s.metadata]
    used, rows, notes = {}, [], []
    for it in chosen:
        dname = it.get("file", "")
        # candidates that share the dataset's study number (or, if none, any) and, if the dataset names a
        # location, that location too; then pick the most similar name.
        d = _name_features(dname)
        scored = []
        for p, st in all_meta:
            s = _name_score(dname, Path(p).name)
            own = (st == it["study"])
            scored.append((s + (1.0 if own else 0.0), own, p, st))
        scored.sort(key=lambda x: (-x[0], -int(x[1])))
        best = scored[0] if scored else (0.0, False, None, None)
        ids = ids_of(it)
        if best[2] is None:
            notes.append(f"[metadata] {dname}: no metadata file found for it")
        else:
            used.setdefault(best[2], best[3])
            n_hit = len(ids & values_of(best[2])) if ids else 0
            if ids and n_hit < len(ids):
                notes.append(f"[metadata] {dname}: metadata file {Path(best[2]).name} holds "
                             f"{n_hit} of its {len(ids)} sample ids ({len(ids) - n_hit} have no metadata row)")
        own_hit = sum(1 for p, _ in all_meta if p == best[2] and ids and any(i in values_of(p) for i in ids))
        rows.append({"study": it["study"], "data file": dname,
                     "metadata files": Path(best[2]).name if best[2] else "none",
                     "sample ids found": f"{len([1 for i in ids if i in values_of(best[2])])} of {len(ids)}"
                     if (ids and best[2]) else ("-" if not ids else "0 of " + str(len(ids)))})
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
        df = pd.DataFrame()
        # NGID workbooks (multi-sheet) hold their sample rows on the 'Sample - NGID' sheet
        try:
            hexmap = _hex_maps(path)
            df, nnotes = read_ngid_metadata(path, folder_study, hexmap)
            notes += nnotes
        except Exception:
            df = pd.DataFrame()
        if df.empty:
            try:
                raw = read_full(str(path))
            except Exception as exc:
                notes.append(f"[metadata] {Path(path).name}: cannot read ({exc})")
                continue
            df = _std_columns(raw).astype(object)
        else:
            df = df.astype(object)
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
    # Clinical reference: canonical study attribution (dual-enrolled patients), subject, group,
    # stage and severity. The NGID sheets carry patient/site/lesion/visit but not the scores; the
    # reference (meta_data_all_complete.csv) has them in the same DS##NBD## id space. Where a sample
    # is in the reference it is attributed to the reference's study, so a sample that appears in two
    # study folders (a shared sequencing run) lands once, in its true study.
    combined, ref_notes = _apply_reference(combined, [str(p) for p, _ in files])
    notes += ref_notes
    flags = _patient_level_flags(combined)
    disc = pd.DataFrame(flags, columns=["kind", "study", "item", "field", "used", "values", "files", "rule"])
    n_multi = int((allrows.groupby("_key")["_file"].nunique() > 1).sum())
    notes.append(f"[metadata] combined {len(allrows)} rows of {len(frames)} file(s) into {len(combined)} samples"
                 + (f"; {n_multi} sample(s) are in more than one file" if n_multi else "")
                 + (f"; {len(disc)} discrepancies (see the list on Data setup)" if len(disc) else "; no discrepancies"))
    # 1806 has no MF patients: any MF group on a 1806 row is an artifact (a leak from the
    # reference-enrichment of a shared subject number). Clear it to the patient's actual group.
    mf_1806 = (combined["study"].astype(str).str.strip() == "1806") & \
              (combined["group"].astype(str).str.strip().str.lower() == "mf")
    if mf_1806.any():
        for subj in combined.loc[mf_1806, "subject"].unique():
            others = combined[(combined["subject"].astype(str) == subj) &
                              (combined["group"].astype(str).str.strip().str.lower().isin(["psoriasis", "healthy"]))]
            fix = others["group"].iloc[0] if len(others) else "psoriasis"
            combined.loc[mf_1806 & (combined["subject"].astype(str) == subj), "group"] = fix
            combined.loc[mf_1806 & (combined["subject"].astype(str) == subj), "stage"] = ""
        notes.append(f"[metadata] {int(mf_1806.sum())} 1806 row(s) had a spurious MF group "
                     f"(1806 has no MF patients); set to the patient's actual group")
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


# =============================================================== clinical reference enrichment
def _find_reference() -> Path | None:
    """The clinical reference table (meta_data_all_complete.csv or a flat metadata_all csv) anywhere
    above the current data drive / up the workspace tree. It is the canonical source of study,
    subject, group, stage and severity for the microbiome samples."""
    here = Path(__file__).resolve().parent
    candidates = [
        "meta_data_all/meta_data_all_complete.csv",
        "meta_data_all/meta_data_all3_new.csv",
        "meta_data_all_complete.csv",
        "meta_data_all3_new.csv",
    ]
    cur = here
    for _ in range(8):
        for c in candidates:
            p = cur / c
            if p.is_file():
                return p
        if cur == cur.parent:
            break
        cur = cur.parent
    return None


def _apply_reference(combined: pd.DataFrame, file_paths: list) -> tuple[pd.DataFrame, list]:
    """Enrich the combined metadata with the clinical reference (canonical study, subject, group,
    stage, severity). Returns (combined, notes). No-op (empty notes) when no reference is found."""
    notes = []
    ref_path = _find_reference()
    if ref_path is None:
        notes.append("[metadata] no clinical reference (meta_data_all_complete.csv) found above the data "
                     "drive: group / stage / severity are taken from the NGID sheets' diagnosis / "
                     "lesion_type only")
        return combined, notes
    try:
        ref = pd.read_csv(ref_path, dtype=str)
    except Exception as exc:
        notes.append(f"[metadata] clinical reference {ref_path.name} could not be read ({exc})")
        return combined, notes
    ref.columns = [re.sub(r"[\s\-]+", "_", str(c).strip().lower()) for c in ref.columns]
    ref = ref.dropna(subset=["sample_id"]) if "sample_id" in ref else ref
    if "sample_id" not in ref or ref.empty:
        notes.append("[metadata] clinical reference has no sample_id column; not used")
        return combined, notes
    by_id = {}
    for _, r in ref.iterrows():
        sid = str(r["sample_id"]).strip().lower()
        if sid:
            by_id.setdefault(sid, r)
    study_of, subj_of, group_of, stage_of, gp_of, ls_of, ts_of, rls_of, region_of, batch_of = (
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {})
    for sid, r in by_id.items():
        study_of[sid] = str(r.get("study", "")).strip()
        subj_of[sid] = str(r.get("subject", "")).strip()
        group_of[sid] = str(r.get("group", "")).strip()
        stage_of[sid] = str(r.get("stage", "")).strip()
        gp_of[sid] = str(r.get("global_severity", "")).strip()
        ls_of[sid] = str(r.get("local_severity", "")).strip()
        ts_of[sid] = str(r.get("timepoint", "")).strip()
        rls_of[sid] = str(r.get("lesion_status", "")).strip()
        region_of[sid] = str(r.get("region_location", "") or r.get("region", "")).strip()
        batch_of[sid] = str(r.get("batch", "")).strip()

    if "sample_id" not in combined:
        return combined, notes
    sids = combined["sample_id"].astype(str).str.strip().str.lower()
    n = int(sids.isin(by_id).sum())
    # canonical study: where the reference says a sample belongs to another study (dual-enrolled),
    # re-attribute it and drop the duplicate that the NGID sheets produced in the other folder.
    ref_study = sids.map(lambda x: study_of.get(x, ""))
    changed_study = ref_study.ne("") & (ref_study != combined["study"])
    combined["study"] = combined["study"].mask(changed_study, ref_study[changed_study])
    if changed_study.any():
        before = len(combined)
        combined = combined.drop_duplicates(subset=["study", "sample_id"], keep="last").reset_index(drop=True)
        notes.append(f"[metadata] {int(changed_study.sum())} dual-enrolled sample(s) re-attributed to the "
                     f"clinical reference's study; {before - len(combined)} duplicate row(s) removed")
    # subject / group / stage / severity from the reference (the sheet's value is kept when the reference
    # has no row for that sample, or no value)
    def fill(col, src_map, keep_blank_only=True):
        if col not in combined:
            combined[col] = ""
        m = sids.map(lambda x: src_map.get(x, ""))
        if keep_blank_only:
            combined[col] = combined[col].mask((combined[col].astype(str).str.strip() == "") & m.ne(""), m)
        else:
            combined[col] = m.mask(m.eq(""), combined[col])
    subj_new = sids.map(lambda x: subj_of.get(x, ""))
    keep_subj = combined["subject"].astype(str).str.strip().eq("")
    combined["subject"] = combined["subject"].mask(keep_subj & subj_new.ne(""), subj_new)
    fill("group", group_of)
    # STAGE: the NGID sheet's per-sample diagnosis is the authority (a patient's stage may differ over
    # time and across lesions, and the sequencing metadata is more correct than the flat reference).
    # Keep each sample's own stage; only fill it from the reference where the sheet left it blank.
    fill("stage", stage_of)
    # healthy patients have no MF stage
    healthy = combined["group"].astype(str).str.strip().str.lower().isin(["healthy", "hv"])
    combined["stage"] = combined["stage"].where(~healthy, "")
    fill("global_severity", gp_of)
    fill("local_severity", ls_of)
    if "lesion_status" in combined:
        ls_new = sids.map(lambda x: rls_of.get(x, ""))
        combined["lesion_status"] = combined["lesion_status"].mask(
            combined["lesion_status"].astype(str).str.strip().eq("") & ls_new.ne(""), ls_new)
    if "region_location" in combined:
        rg_new = sids.map(lambda x: region_of.get(x, ""))
        combined["region_location"] = combined["region_location"].mask(
            combined["region_location"].astype(str).str.strip().eq("") & rg_new.ne(""), rg_new)
    if "batch" in combined:
        bt_new = sids.map(lambda x: batch_of.get(x, ""))
        combined["batch"] = combined["batch"].mask(
            combined["batch"].astype(str).str.strip().eq("") & bt_new.ne(""), bt_new)
    notes.append(f"[metadata] clinical reference {ref_path.name}: {n} of {len(combined)} samples matched "
                 f"(study / subject / group / stage / severity filled from it)")
    return combined, notes


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
