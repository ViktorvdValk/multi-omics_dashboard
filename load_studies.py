"""Load filtered Emu tables for studies 2230, 1947, and 1806 (skin, nostril and gut samples)."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import skin_engine as E


DATA_ROOT = Path("/home/vovandervalk/data1/sequencing")
FILTERED_SUFFIX = "_rel-abundance_filtered.tsv"

# Every study maps to a LIST of Emu output folders. The faecal (FC) samples of 1947 are assumed to sit
# in the shared 1806 folder; a metadata sample without a matching TSV is reported in the notes, so a
# wrong assumption shows up immediately. Add folders here if gut samples live elsewhere.
SHARED_SK_FC = Path("CHDR1806_SK_FC_and_CHDR1947_FC/output_emu_99_emu_db")
STUDIES = {
    "2230": [Path("CHDR2230_SK/output_emu_99_emu_db")],
    "1947": [Path("CHDR1947_SK/output_emu_99_emu_db"), SHARED_SK_FC],
    "1806": [SHARED_SK_FC],
}

# Order of ALL severity labels from mildest (blue) to most severe (red), lower-case, as they appear after
# loading (MF stages from the 'stage' column, psoriasis mild/moderate/severe derived from PASI). 'healthy' is
# always green and first. Labels you leave out are appended alphabetically. Example (adapt to your labels):
#   SEVERITY_ORDER = ["ia", "ib", "iia", "iib", "iiia", "iva", "ivb", "sezary", "mild", "moderate", "severe"]
SEVERITY_ORDER: list[str] = []
E.SEVERITY_ORDER = SEVERITY_ORDER

# A patient can have more than one lesion sampled at a timepoint (study 2230). Which one is the most severe is
# read from the 'local_severity' column of the metadata (LESION_COLUMN). If that column is absent, one of
# LESION_COLUMN_NAMES is used, and otherwise
# for a suffix in lesion_status ('lesional_1', 'lesional severe', 'lesional_tumor' ...).
# LESION_ORDER: values of that column from MOST to LEAST severe, e.g. ["tumor", "plaque", "patch"].
# Left empty, known words are used (tumor > plaque > patch, severe > moderate > mild, most > less severe);
# numbers count as a severity score (higher = more severe) in a *severity / *score / *grade column and as a
# lesion number (1 = most severe) otherwise. The rule that was used is reported in the loading notes.
LESION_COLUMN: str | None = "local_severity"
LESION_ORDER: list[str] = []
LESION_COLUMN_NAMES = ("local_severity", "lesion_severity", "lesion_score", "lesion_grade", "lesion_rank", "lesion_order",
                       "lesion_type", "lesion_nr", "lesion_number", "lesion_id", "lesion")
_LESION_WORDS = {"most severe": 3, "most_severe": 3, "severe": 3, "worst": 3, "primary": 3, "main": 3, "index": 3,
                 "tumor": 3, "tumour": 3, "moderate": 2, "plaque": 2, "less severe": 1, "less_severe": 1,
                 "least severe": 1, "mild": 1, "secondary": 1, "other": 1, "patch": 1}


def _lesion_scores(values: pd.Series, column: str) -> tuple[pd.Series, str]:
    """Severity score per value (higher = more severe) and a description of the rule used."""
    text = values.astype(object).map(lambda v: np.nan if pd.isna(v) or str(v).strip() == "" else str(v).strip().lower())
    if LESION_ORDER:
        order = [str(v).strip().lower() for v in LESION_ORDER]
        return text.map(lambda v: len(order) - order.index(v) if v in order else np.nan), f"LESION_ORDER {order}"
    numbers = pd.to_numeric(text.map(lambda v: v.replace(",", ".") if isinstance(v, str) else v), errors="coerce")
    if text.notna().any() and numbers[text.notna()].notna().all():                 # decimal comma or point
        if re.search(r"sever|score|grade", column):
            return numbers, f"'{column}': higher number = more severe"
        return -numbers, f"'{column}': lesion 1 = most severe"
    known = text.map(lambda v: _LESION_WORDS.get(v, np.nan))
    unknown = sorted(set(text.dropna()) - set(_LESION_WORDS))
    rule = f"'{column}': " + " > ".join(sorted(set(text.dropna()) & set(_LESION_WORDS), key=lambda v: -_LESION_WORDS[v]))
    if unknown:
        rule += f" (values not understood: {unknown[:5]}; set LESION_ORDER in load_studies.py)"
    return known, rule


def local_severity_values(rows: pd.DataFrame, site: pd.Series) -> pd.Series:
    """The LESION_COLUMN value of every lesional sample as a number (decimal comma allowed); words are scored
    like the lesion ranking (LESION_ORDER, or severe 3 > moderate 2 > mild 1). NaN for other sites."""
    column = LESION_COLUMN.lower() if LESION_COLUMN else None
    if not column or column not in rows:
        return pd.Series(np.nan, index=rows.index)
    text = rows[column].astype(object).map(lambda v: np.nan if pd.isna(v) or str(v).strip() == ""
                                           else str(v).strip().replace(",", "."))
    values = pd.to_numeric(text, errors="coerce")
    if text.notna().any() and values[text.notna()].isna().any():           # words, not numbers
        values, _ = _lesion_scores(rows[column], column)
    return values.where(site.eq("lesional").values)


def rank_lesions(rows: pd.DataFrame, subjects: pd.Series, site: pd.Series, study: str):
    """lesion_rank (1 = most severe of the patient's lesions at that timepoint) and a label per sample.
    Returns (site with suffixes removed, rank, label, notes)."""
    notes = []
    site = site.copy()
    scores = pd.Series(np.nan, index=rows.index)
    rule = None
    column = LESION_COLUMN.lower() if LESION_COLUMN and LESION_COLUMN.lower() in rows else None
    column = column or next((c for c in LESION_COLUMN_NAMES if c in rows), None)
    if column and column in rows:
        scores, rule = _lesion_scores(rows[column], column)
    suffix = site.str.extract(r"^(?:lesional|lesion|l)[\s_\-:]+(.+)$", expand=False)
    has_suffix = suffix.notna() & ~site.str.startswith("non")
    if has_suffix.any():
        site = site.mask(has_suffix, "lesional")
        if rule is None:
            scores, rule = _lesion_scores(suffix.where(has_suffix), "lesion_status suffix")
    tp = (rows["timepoint"] if "timepoint" in rows else pd.Series("", index=rows.index)).astype(object).fillna("")
    lesional = site.eq("lesional")
    key = [subjects.astype(str).values, tp.astype(str).values]
    count = lesional.astype(int).groupby(key).transform("sum")
    multi = lesional & (count > 1)
    rank = pd.Series(np.nan, index=rows.index)
    rank[lesional] = 1.0
    label = pd.Series("", index=rows.index, dtype=object)
    if multi.any():
        sc = scores.where(multi)
        rank[multi] = sc[multi].groupby([subjects[multi].astype(str).values, tp[multi].astype(str).values]).rank(
            ascending=False, method="min").fillna(1)
        label[multi] = np.where(rank[multi] == 1, "most severe lesion", "less severe lesion")
        n_pat = subjects[multi].nunique()
        unranked = multi & sc.isna()
        if rule is None or unranked.all():
            notes.append(f"[{study}] {n_pat} patient(s) have more than one lesional sample at a timepoint, but the "
                         f"metadata has no column saying which lesion is the most severe; their lesions are averaged. "
                         f"Set LESION_COLUMN (and LESION_ORDER) in load_studies.py.")
            rank[multi], label[multi] = 1.0, "lesion (severity unknown)"
        else:
            notes.append(f"[{study}] {n_pat} patient(s) have more than one lesional sample at a timepoint; the most "
                         f"severe is read from {rule}"
                         + (f". {int(unranked.sum())} lesion(s) without a value count as most severe"
                            if unranked.any() else ""))
    return site, rank, label, notes


# Values of the metadata 'site' column that are analysed ('skin' rows are split further by
# 'lesion_status' into lesional / non-lesional / nostril; 'gut' rows become site 'gut').
COMPARTMENTS = {"skin", "gut"}


def find_metadata(data_root: Path, metadata_path: str | Path | None) -> Path:
    if metadata_path:
        path = Path(metadata_path)
        return path if path.is_absolute() else data_root / path

    for filename in ("meta_data_all_complete.csv", "meta_data_all3_new.csv", "metadata_all3_new.csv"):
        path = data_root / filename
        if path.is_file():
            return path

    raise FileNotFoundError(
        "Could not find meta_data_all_complete.csv, meta_data_all3_new.csv or metadata_all3_new.csv "
        f"under {data_root}"
    )


def load_filtered_rank(
    results_dirs: list[Path],
    rank: str,
    allowed_samples: set[str],
    known_samples: set[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Samples x taxa relative abundance from the filtered Emu TSVs in one or more folders.

    `allowed_samples` are the samples to load; `known_samples` (any study/site in the metadata) are
    ignored silently when found, so shared folders do not produce noise. Returns (matrix, n_unknown)."""
    known_samples = known_samples or set()
    files: list[Path] = []
    for results_dir in results_dirs:
        if not results_dir.is_dir():
            raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
        found = sorted(results_dir.rglob(f"*{FILTERED_SUFFIX}"))
        if not found:
            raise FileNotFoundError(
                f"No *{FILTERED_SUFFIX} files found under {results_dir}"
            )
        files.extend(found)

    columns: dict[str, pd.Series] = {}
    origin: dict[str, Path] = {}
    ignored = 0

    for path in files:
        sample = path.name[: -len(FILTERED_SUFFIX)]
        if sample not in allowed_samples:
            if sample not in known_samples:
                ignored += 1
            continue
        if sample in columns:
            raise ValueError(
                f"Multiple filtered TSV files found for {sample!r}: {origin[sample]} and {path}"
            )

        df = pd.read_csv(path, sep="\t")
        required = {"tax_id", "abundance", rank}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{path} is missing required column(s): "
                f"{', '.join(sorted(missing))}"
            )

        if "keep_for_analysis" in df.columns:
            keep = df["keep_for_analysis"].astype(str).str.strip().str.lower()
            df = df[keep.isin({"true", "1", "yes", "y"})]

        df = df[
            df["tax_id"].astype(str).str.strip().str.lower() != "unassigned"
        ].copy()
        df["abundance"] = pd.to_numeric(df["abundance"], errors="raise")

        abundance = df["abundance"].to_numpy(dtype=float)
        if not np.isfinite(abundance).all() or (abundance < 0).any():
            raise ValueError(f"{path} contains invalid or negative abundance values")

        taxa = df[rank].fillna("Unclassified").astype(str)
        by_taxon = df.groupby(taxa)["abundance"].sum()
        if by_taxon.sum() <= 0:
            raise ValueError(
                f"{path} has no positive abundance after filtering"
            )
        columns[sample] = by_taxon
        origin[sample] = path

    if not columns:
        raise FileNotFoundError(
            "No filtered TSV sample IDs matched the metadata in "
            f"{', '.join(str(d) for d in results_dirs)}"
        )

    mat = pd.DataFrame(columns).fillna(0.0).T
    mat.index.name = "sample"
    return mat.div(mat.sum(axis=1), axis=0), ignored


# --------------------------------------------------------------- BIOM input (output of split_biom.py)
# The microbiome can also be read from BIOM files (e.g. one per study x site, made with emu_to_biom.py and
# split_biom.py). The BIOM folder is absolute or relative to the data root (blank: <data root>/BIOM_DIR if it
# exists, else the data root itself); every *.biom file in it and its subfolders is read.
# A metadata sample_id is matched to a BIOM sample by the BIOM sample ID or by the Emu file the sample came from
# (the 'source_file' that emu_to_biom.py stores), with or without folder, '.tsv' and '_rel-abundance...' suffix.
BIOM_DIR = Path("biom")
SOURCES = {"tsv": "filtered Emu TSVs", "biom": "BIOM files", "both": "filtered Emu TSVs + BIOM files"}
_RANK_POSITION = {"superkingdom": 0, "kingdom": 0, "domain": 0, "phylum": 1, "class": 2, "order": 3,
                  "family": 4, "genus": 5, "species": 6}
_RANK_PREFIX = {"k": 0, "d": 0, "p": 1, "c": 2, "o": 3, "f": 4, "g": 5, "s": 6}


def _strip_emu_suffix(name: str) -> str:
    """'S01_rel-abundance_filtered.tsv' / 'S01_rel-abundance.tsv' / 'S01' -> 'S01'."""
    return re.sub(r"_rel-abundance.*$", "", re.sub(r"\.tsv$", "", str(name).strip()))


def _biom_sample_keys(sample_id: str, sample_md) -> set[str]:
    keys = {sample_id, _strip_emu_suffix(sample_id)}
    src = str((sample_md or {}).get("source_file", "") or "").strip()
    if src:
        src = src.replace("\\", "/")
        name = src.rsplit("/", 1)[-1]
        parent = src.rsplit("/", 2)[-2] if src.count("/") else ""
        keys |= {src, name, _strip_emu_suffix(name), f"{parent}/{name}"}
    keys |= {k.lower() for k in keys}                      # case is not used to tell samples apart
    return {k for k in keys if k}


_COMPARTMENT_WORDS = {"skin": "skin", "nostril": "skin", "nasal": "skin", "nose": "skin", "rnos": "skin",
                      "gut": "gut", "faecal": "gut", "fecal": "gut", "feces": "gut", "faeces": "gut", "stool": "gut"}


def _study_hint(text) -> str | None:
    m = re.search(r"(?<!\d)(\d{4})(?!\d)", str(text or ""))
    return m.group(1) if m else None


def _compartment_hint(text) -> str | None:
    words = re.findall(r"[a-z]+", str(text or "").lower())
    return next((_COMPARTMENT_WORDS[w] for w in words if w in _COMPARTMENT_WORDS), None)


def resolve_biom_dir(root: Path, biom_dir) -> Path:
    """The BIOM folder (or one .biom file). Given: absolute, or relative to the data root (else to the working
    folder). Blank: the 'biom' folder in the data root if there is one, else the data root itself."""
    if biom_dir:
        p = Path(str(biom_dir).strip().strip('"'))
        tries = [p] if p.is_absolute() else [root / p, p]
    else:
        tries = [root / BIOM_DIR, root]
    for p in tries:
        if p.is_dir() or (p.is_file() and p.suffix.lower() == ".biom"):
            return p
    raise FileNotFoundError(f"BIOM folder not found; tried {', '.join(str(p) for p in tries)}")


def read_biom_folder(biom_dir: Path) -> dict:
    """Counts, taxonomy and sample keys of every *.biom file below biom_dir (or of one .biom file), read once per
    load. A sample is identified by file + sample ID, so the same ID in two files (e.g. a barcode reused in
    another study) is fine. The study and compartment of a sample come from its sample metadata ('study', 'site',
    as written by split_biom.py) or else from the file name (e.g. emu_1806_gut.biom)."""
    try:
        from biom import load_table
    except ImportError as exc:
        raise ImportError("BIOM input needs the biom-format package: pip install biom-format h5py") from exc
    biom_dir = Path(biom_dir)
    if biom_dir.is_file():
        files = [biom_dir]
    elif biom_dir.is_dir():
        files = sorted(biom_dir.rglob("*.biom"))
    else:
        raise FileNotFoundError(f"BIOM folder does not exist: {biom_dir}")
    if not files:
        raise FileNotFoundError(f"No *.biom files found in {biom_dir} or its subfolders")

    counts: dict[str, pd.Series] = {}
    info: dict[str, dict] = {}
    taxonomy: dict[str, list[str]] = {}
    keys: dict[str, set[str]] = {}
    for path in files:
        try:
            table = load_table(str(path))
        except Exception as exc:
            raise ValueError(f"cannot read BIOM file {path}: {exc}") from exc
        df = table.to_dataframe(dense=True)
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)
        for oid, m in zip(df.index, table.metadata(axis="observation") or [None] * len(df)):
            lineage = (m or {}).get("taxonomy")
            if isinstance(lineage, str):
                lineage = [t.strip() for t in lineage.split(";")]
            if lineage is not None:
                taxonomy.setdefault(oid, [str(t) for t in lineage])
        file_study, file_comp = _study_hint(path.stem), _compartment_hint(path.stem)
        for sid, m in zip(df.columns, table.metadata(axis="sample") or [None] * df.shape[1]):
            uid = f"{path}::{sid}"
            counts[uid] = df[sid]
            info[uid] = {"sid": sid, "file": path,
                         "study": _study_hint((m or {}).get("study")) or file_study,
                         "compartment": _compartment_hint((m or {}).get("site")) or file_comp}
            for k in _biom_sample_keys(sid, m):
                keys.setdefault(k, set()).add(uid)
    return {"dir": biom_dir, "files": files, "counts": counts, "info": info, "taxonomy": taxonomy, "keys": keys}


def _biom_taxon(oid: str, lineage: list[str] | None, rank: str) -> str | None:
    """Name of a BIOM feature at `rank` (None for unassigned reads)."""
    if oid.strip().lower() == "unassigned" or (lineage and lineage[0].strip().lower() == "unassigned"):
        return None
    if not lineage:
        return oid if rank == "species" else "Unclassified"
    want = _RANK_POSITION[rank]
    prefixed = [t for t in lineage if re.match(r"^[a-z]__", t.strip())]
    if prefixed:
        hit = [t for t in prefixed if _RANK_PREFIX.get(t.strip()[0]) == want]
        name = hit[0].strip()[3:] if hit else ""
    else:
        name = lineage[want].strip() if want < len(lineage) else ""
    return name if name and name.lower() not in ("nan", "none") else "Unclassified"


def load_biom_rank(biom: dict, rank: str, samples, study: str
                   ) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Samples x taxa relative abundance of the metadata samples found in the BIOM files (index = metadata
    sample_id); every taxon in the BIOM file is used. `samples`: {metadata sample_id: site} (or a set of ids).
    Only BIOM samples of this study and of the sample's compartment (skin / gut) are candidates when the BIOM
    file says which (sample metadata or file name). Returns (matrix, {metadata sample: BIOM sample}, notes)."""
    if rank not in _RANK_POSITION:
        raise ValueError(f"Rank {rank!r} cannot be read from BIOM taxonomy; use one of {sorted(_RANK_POSITION)}")
    sites = samples if isinstance(samples, dict) else {s: None for s in samples}
    columns: dict[str, pd.Series] = {}
    used: dict[str, str] = {}
    ambiguous, empty = [], []
    n_unclassified = 0
    for sample in sorted(sites):
        name = Path(str(sample).replace("\\", "/")).name
        hits = set()
        for k in (sample, _strip_emu_suffix(name), str(sample).lower(), _strip_emu_suffix(name).lower()):
            hits = biom["keys"].get(k, set())
            if hits:
                break
        comp = None if sites[sample] is None else ("gut" if sites[sample] == "gut" else "skin")
        hits = {h for h in hits if biom["info"][h]["study"] in (None, study)
                and (comp is None or biom["info"][h]["compartment"] in (None, comp))}
        if not hits:
            continue
        if len(hits) > 1:
            ambiguous.append(f"{sample} ({', '.join(sorted(Path(biom['info'][h]['file']).name for h in hits))})")
            continue
        uid = next(iter(hits))
        if uid in used.values():
            raise ValueError(f"[{study}] BIOM sample {biom['info'][uid]['sid']!r} matches more than one metadata "
                             f"sample_id ({[s for s, b in used.items() if b == uid] + [sample]})")
        col = biom["counts"][uid]
        col = col[col > 0]
        pairs = [(_biom_taxon(o, biom["taxonomy"].get(o), rank), v) for o, v in col.items()]
        pairs = [(t, v) for t, v in pairs if t is not None]
        if not pairs:
            empty.append(sample)
            continue
        by_taxon = pd.Series([v for _, v in pairs], index=[t for t, _ in pairs], dtype=float).groupby(level=0).sum()
        n_unclassified += int(set(by_taxon.index) == {"Unclassified"})
        columns[sample] = by_taxon
        used[sample] = uid
    notes = []
    if ambiguous:
        notes.append(f"[{study}] {len(ambiguous)} metadata sample(s) match a sample in more than one BIOM file of this "
                     f"study and site and were skipped: {ambiguous[:5]}")
    if empty:
        notes.append(f"[{study}] {len(empty)} BIOM sample(s) without assigned reads skipped: {empty[:10]}")
    if columns and n_unclassified == len(columns):
        notes.append(f"[{study}] every BIOM sample is 'Unclassified' at rank '{rank}': the BIOM taxonomy has no "
                     f"{rank} level (made with emu_to_biom.py --rank above {rank}?)")
    if not columns:
        return pd.DataFrame(), used, notes
    mat = pd.DataFrame(columns).fillna(0.0).T
    mat.index.name = "sample"
    return mat.div(mat.sum(axis=1), axis=0), used, notes


# --------------------------------------------------------------- Biotaviz tables (V2: files on the data drive)
# A Biotaviz export is a tree: one row per taxon at every level, numbered 1. / 1.1. / 1.1.1. ..., the name written
# as "<rank> - <name>" ("no - Root", "phylum - Bacillota", "genus - Cutibacterium"), one column per sample with read
# counts. A higher level is the SUM of the levels below it, so only the rows of the chosen rank are used; the reads
# of the root that are not assigned down to that rank become "Unclassified" (as for the Emu tables).
_BIOTAVIZ_NAME = re.compile(r"^\s*([A-Za-z_ ]+?)\s+-\s+(.*\S)\s*$")
_BIOTAVIZ_ID = re.compile(r"^\s*\d+(\.\d+)*\.?\s*$")


def looks_like_biotaviz(frame: pd.DataFrame) -> bool:
    """frame: the first rows of a table read without header."""
    if frame.shape[1] < 3:
        return False
    first = frame.iloc[:, 0].astype(str).str.strip().str.lower()
    if first.str.startswith("#class").any():
        return True
    ids = frame.iloc[:, 0].astype(str).str.match(_BIOTAVIZ_ID)
    names = frame.iloc[:, 1].astype(str).str.match(_BIOTAVIZ_NAME)
    return int((ids & names).sum()) >= 3


def read_biotaviz(raw: pd.DataFrame, rank: str, label: str = "") -> tuple[pd.DataFrame, list[str]]:
    """raw: the sheet read without header (all text). Returns (samples x taxa relative abundance at `rank`, notes)."""
    if rank not in _RANK_POSITION:
        raise ValueError(f"Rank {rank!r} is not a Biotaviz level; use one of {sorted(_RANK_POSITION)}")
    notes = []
    first = raw.iloc[:, 0].astype(str).str.strip()
    head = first.str.lower().str.startswith("#class")
    if head.any():
        h = int(np.flatnonzero(head.values)[0])
    else:                                              # no header row: the row before the first numbered row
        h = max(int(np.flatnonzero(first.str.match(_BIOTAVIZ_ID).values)[0]) - 1, 0)
    header = raw.iloc[h].astype(str).str.strip().tolist()
    body = raw.iloc[h + 1:].reset_index(drop=True)
    body = body[body.iloc[:, 0].astype(str).str.match(_BIOTAVIZ_ID) & body.iloc[:, 1].notna()]
    samples = [c for c in header[2:] if c and c.lower() not in ("nan", "none")]
    cols = list(range(2, 2 + len(samples)))
    parsed = body.iloc[:, 1].astype(str).str.extract(_BIOTAVIZ_NAME)
    level, name = parsed[0].str.strip().str.lower(), parsed[1].str.strip()
    values = body.iloc[:, cols].apply(lambda c: pd.to_numeric(c.astype(str).str.replace(",", ".", regex=False),
                                                              errors="coerce")).fillna(0.0)
    values.columns = samples
    depth = body.iloc[:, 0].astype(str).str.strip().str.rstrip(".").str.count(r"\.")
    root = values[(depth == 0).values].sum(axis=0)
    if not (root > 0).any():
        raise ValueError(f"{label}: no root row with read counts ('1.  no - Root')")
    at_rank = values[(level == rank).values]
    taxa = name[(level == rank).values].values
    if at_rank.empty:
        raise ValueError(f"{label}: no rows of rank '{rank}' (levels found: {sorted(set(level.dropna()))})")
    mat = at_rank.groupby(taxa).sum().T                            # samples x taxa (reads)
    rest = (root - mat.sum(axis=1)).clip(lower=0)
    if (rest > 0).any():
        mat["Unclassified"] = mat.get("Unclassified", 0) + rest
    empty = root.index[root <= 0]
    if len(empty):
        notes.append(f"{label}: {len(empty)} sample column(s) without reads skipped: {list(empty[:8])}")
    mat = mat.loc[root.index[root > 0]]
    dup = pd.Index(mat.index).duplicated()
    if dup.any():
        notes.append(f"{label}: sample column(s) appear twice, the first is used: {list(mat.index[dup][:8])}")
        mat = mat[~dup]
    mat.index = mat.index.astype(str).str.strip()
    return mat.div(root.loc[mat.index], axis=0), notes


def read_biom_file(path, rank: str) -> tuple[pd.DataFrame, list[str]]:
    """One .biom file -> samples x taxa relative abundance at `rank` (sample = the BIOM sample ID)."""
    biom = read_biom_folder(Path(path))
    cols = {}
    for uid, col in biom["counts"].items():
        col = col[col > 0]
        pairs = [(_biom_taxon(o, biom["taxonomy"].get(o), rank), v) for o, v in col.items()]
        pairs = [(t, v) for t, v in pairs if t is not None]
        if pairs:
            cols[biom["info"][uid]["sid"]] = pd.Series([v for _, v in pairs], index=[t for t, _ in pairs]).groupby(level=0).sum()
    if not cols:
        return pd.DataFrame(), [f"{Path(path).name}: no sample with assigned reads"]
    mat = pd.DataFrame(cols).fillna(0.0).T
    return mat.div(mat.sum(axis=1), axis=0), []


def assemble_microbiome(tables: list, metadata_file
                        ) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame, dict]:
    """tables: [(samples x taxa relative abundance, {'file': name, 'study': study of its folder,
    'compartment': 'skin' | 'gut' | None from the file name})]. The samples are matched to the metadata file by
    sample_id; the study and site of a sample come from the metadata. Returns (matrix, metadata, notes,
    discrepancies) like load_studies, plus a table of the samples whose file contradicts the metadata, and
    {metadata sample_id: file it was read from}."""
    notes, flags = [], []
    metadata_source = pd.read_csv(metadata_file, dtype=str, keep_default_na=False)
    metadata_source.columns = [str(c).strip().lower() for c in metadata_source.columns]
    for c in ("sample_id", "site", "study"):
        if c not in metadata_source:
            metadata_source[c] = ""
    metadata_source = metadata_source.astype(str).apply(lambda col: col.str.strip())
    # skin or gut of a metadata row: its site, else a word in its site / sample source ('skin swab', 'faeces' ...)
    comp = metadata_source["site"].str.lower()
    hint = (metadata_source["site"] + " " + metadata_source.get("sample_source", "")).str.lower()
    other = hint.str.contains(r"blister|plasma|serum|blood")                # energy rows are never skin or gut
    comp = comp.where(comp.isin(COMPARTMENTS) | other,
                      np.where(hint.str.contains(r"faec|fec|stool|gut|\bfc\b"), "gut",
                               np.where(hint.str.contains(r"skin|swab|nasal|nostril|lesion"), "skin", "")))
    comp = comp.where(~other | comp.isin(COMPARTMENTS), "")
    raw_site = metadata_source["site"].copy()
    metadata_source["site"] = comp
    tables = [(mat, info) for mat, info in tables if not mat.empty]
    all_samples = {str(x).strip().lower() for mat, _ in tables for x in mat.index}

    def lookup_on(col):
        out = {}
        for i, v in metadata_source[col].items():
            if v and v.lower() in all_samples:
                out.setdefault(v.lower(), []).append(i)
        return out
    by_col = {"sample_id": lookup_on("sample_id")}
    missing = all_samples - set(by_col["sample_id"])
    if missing:             # the sample names of a file may be in another metadata column (sequencing id ...)
        best = max(((c, lookup_on(c)) for c in metadata_source.columns
                    if c not in ("sample_id", "study", "subject", "site") and not c.startswith("_")),
                   key=lambda cl: len(set(cl[1]) & missing), default=(None, {}))
        if best[0] and set(best[1]) & missing:
            by_col[best[0]] = best[1]
            notes.append(f"{len(set(best[1]) & missing)} microbiome sample name(s) are not a metadata sample_id but are "
                         f"in the metadata column '{best[0]}': matched on that column")
            fill = [i for v, rows in best[1].items() if v in missing for i in rows
                    if not metadata_source.at[i, "sample_id"]]
            for i in fill:                                      # a row without sample_id gets the sample name
                metadata_source.at[i, "sample_id"] = metadata_source.at[i, best[0]]
    lookup = {}
    for col, found in by_col.items():
        for v, rows in found.items():
            if col != "sample_id" and v in by_col["sample_id"]:
                continue
            for i in rows:
                lookup.setdefault(v, []).append((i, metadata_source.at[i, "sample_id"],
                                                 metadata_source.at[i, "study"], metadata_source.at[i, "site"]))
    parts, origin = [], {}
    for mat, info in tables:
        rename, unknown = {}, []
        for sample in mat.index:
            hits = lookup.get(str(sample).strip().lower(), [])
            if hits and info.get("compartment"):          # rows without skin / gut: the file's part decides
                for i, _, _, c in hits:
                    if not c:
                        metadata_source.at[i, "site"] = info["compartment"]
                hits = [(i, sid, st, c or info["compartment"]) for i, sid, st, c in hits]
            no_site = [h for h in hits if h[3] not in COMPARTMENTS]
            hits = [h[1:] for h in hits if h[3] in COMPARTMENTS]
            if no_site and not hits:
                flags.append({"kind": "microbiome sample", "study": no_site[0][2], "item": sample, "field": "site",
                              "used": "", "values": f"metadata site '{raw_site.at[no_site[0][0]]}'",
                              "files": info["file"], "rule": "neither skin nor gut in the metadata or the file name "
                                                             "(set the part on Data setup): not used"})
                continue
            if info.get("study"):
                hits = [h for h in hits if h[1] == info["study"]] or hits
            if not hits:
                unknown.append(sample)
                continue
            if len(hits) > 1:
                flags.append({"kind": "microbiome sample", "study": info.get("study"), "item": sample,
                              "field": "study", "used": "", "values": "; ".join(f"{h[1]} ({h[2]})" for h in hits),
                              "files": info["file"], "rule": "sample id is in more than one study: not used"})
                continue
            sid, study, c = hits[0]
            if info.get("study") and study != info["study"]:
                flags.append({"kind": "microbiome sample", "study": study, "item": sid, "field": "study",
                              "used": study, "values": f"metadata {study}; folder CHDR{info['study']}",
                              "files": info["file"], "rule": "the metadata study is used"})
            if info.get("compartment") and c != info["compartment"]:
                flags.append({"kind": "microbiome sample", "study": study, "item": sid, "field": "site",
                              "used": c, "values": f"metadata {c}; file name suggests {info['compartment']}",
                              "files": info["file"], "rule": "the metadata site is used"})
            if sid in origin:
                flags.append({"kind": "microbiome sample", "study": study, "item": sid, "field": "file",
                              "used": origin[sid], "values": f"{origin[sid]}; {info['file']}",
                              "files": f"{origin[sid]}; {info['file']}", "rule": "in two files: the first is used"})
                continue
            rename[sample] = sid
            origin[sid] = info["file"]
        if unknown:
            notes.append(f"{info['file']}: {len(unknown)} sample(s) are not a skin or gut sample_id of the metadata "
                         f"and were not used, e.g. {list(unknown[:6])}")
        if rename:
            keep = mat.loc[list(rename)]
            keep.index = [rename[s] for s in keep.index]
            parts.append(keep)
    if not parts:
        some = sorted(all_samples)[:4]
        ids = metadata_source.loc[metadata_source["sample_id"].ne(""), "sample_id"].tolist()[:4]
        raise ValueError(f"none of the {len(all_samples)} microbiome sample names (e.g. {some}) is a sample id of "
                         f"a skin or gut row in the metadata used (sample_ids there e.g. {ids}). Check that the "
                         f"study's metadata folder has the metadata of these samples.")
    combined = pd.concat(parts, axis=0).fillna(0.0)
    meta_ids = metadata_source[metadata_source["sample_id"].ne("") & metadata_source["site"].isin(COMPARTMENTS)]
    study_of = dict(zip(meta_ids["sample_id"], meta_ids["study"]))
    metadata_source = metadata_source.apply(E.blank_to_nan)          # blanks -> missing, as read_csv gives
    studies = sorted({study_of[s] for s in combined.index})
    matrices, frames = [], []
    for study in studies:
        md, study_notes = standardize_study_metadata(metadata_source, study)
        notes.extend(study_notes)
        has = md["sample"].isin(combined.index)
        if (~has).any():
            miss = md.loc[~has]
            notes.append(f"[{study}] {int((~has).sum())} metadata sample(s) have no microbiome data in the loaded files "
                         f"(by site: {miss['site'].value_counts().to_dict()}); e.g. {miss['sample'].tolist()[:8]}")
        md = md.loc[has]
        m = combined.loc[md["sample"].tolist()]
        m.index = md.index
        matrices.append(m)
        frames.append(md)
    mat = pd.concat(matrices, axis=0).fillna(0.0)
    mat = mat.loc[:, (mat > 0).any(axis=0)]
    return mat.div(mat.sum(axis=1), axis=0), pd.concat(frames, axis=0), notes, pd.DataFrame(flags), origin


def standardize_study_metadata(
    metadata: pd.DataFrame,
    study: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Skin rows keep their lesion_status as site; gut rows get site 'gut' (no skin type)."""
    study_rows = metadata[
        metadata["study"].astype(str).str.strip() == study
    ]
    compartment_all = study_rows["site"].fillna("").astype(str).str.strip().str.lower()
    rows = study_rows[compartment_all.isin(COMPARTMENTS)].copy()
    compartment = compartment_all.loc[rows.index]

    if rows.empty:
        raise ValueError(f"No skin or gut metadata rows found for study {study}")

    rows["sample_id"] = rows["sample_id"].astype(str).str.strip()
    if rows["sample_id"].duplicated().any():
        duplicates = rows.loc[
            rows["sample_id"].duplicated(keep=False), "sample_id"
        ].unique()
        raise ValueError(
            f"Duplicate sample IDs in study {study}: "
            f"{list(duplicates[:10])}"
        )

    subjects = rows["subject"].fillna("").astype(str).str.strip()
    subjects = subjects.where(subjects.ne(""), rows["sample_id"])

    raw_groups = rows["group"].fillna("").astype(str).str.strip()

    # The disease group is a patient-level attribute. A subject's rows can still disagree (e.g. the only
    # healthy volunteer with faecal data has their gut row written as a disease group): resolve it to ONE
    # group per subject -- a subject's skin rows are authoritative, and 'healthy' beats a disease label --
    # instead of letting the mismatch propagate and failing the whole study. Case is preserved from the file
    # (only lower-cased for the 'healthy' test); 'healthy' itself is the engine's canonical form.
    def one_subject_group(frame):
        vals = [v for v in frame if v and v.lower() != "nan"]
        if not vals:
            return np.nan
        if any(v.lower() in E.HEALTHY_LABELS for v in vals):
            return E.HEALTHY
        key = min(v.lower() for v in vals)
        return next(v for v in vals if v.lower() == key)
    groups = subjects.map(lambda s: one_subject_group(raw_groups[subjects == s].tolist())).astype(object)
    groups = groups.where(groups.notna(), "unspecified").astype(str).str.strip()
    is_psoriasis = groups.str.lower().eq("psoriasis")

    stage = E.blank_to_nan(rows["stage"])          # blank cells -> missing, stays text

    def one_patient_value(values: pd.Series) -> object:
        present = values.dropna().astype(str).str.strip().str.lower().unique()
        if len(present) > 1:
            raise ValueError(
                f"Inconsistent patient-level metadata for study {study}: "
                f"{list(present)}"
            )
        return present[0] if len(present) else np.nan

    # patient-level values are shared by a subject's skin AND gut rows
    patient_stage = stage.groupby(subjects).transform(one_patient_value)

    global_score = pd.to_numeric(rows["global_severity"].astype(object).map(
        lambda v: np.nan if pd.isna(v) else str(v).strip().replace(",", ".")), errors="coerce")
    patient_score = global_score.groupby(subjects).transform("max")
    patient_score = patient_score.where(is_psoriasis)
    # Global severity as a number for every disease group (PASI, mSWAT ...): the score of ONE visit. Every row
    # keeps the value written in it. A row without a value takes the value of the other rows of the same patient
    # at the same timepoint (as written in this file), only when those rows agree; it is never taken from
    # another timepoint. Healthy volunteers: 0.
    tp_raw = rows["timepoint"] if "timepoint" in rows else pd.Series("", index=rows.index)
    tp_key = tp_raw.astype(object).where(tp_raw.notna(), "").astype(str).str.strip()
    by_visit = global_score.groupby([subjects.values, tp_key.values])
    lo, hi = by_visit.transform("min"), by_visit.transform("max")
    visit_score = global_score.where(global_score.notna(), hi.where(np.isclose(lo, hi)))
    visit_score = visit_score.mask(groups.str.lower().isin(E.HEALTHY_LABELS), 0.0)

    lesion_status = rows["lesion_status"].astype("string").str.strip().str.lower()
    lesion_status = lesion_status.replace({
        "rnos": "nostril",
        "non_lesional": "non-lesional",
        "nonlesional": "non-lesional",
    })
    site = lesion_status.mask(compartment.eq("gut"), "gut")
    site, lesion_rank, lesion_label, lesion_notes = rank_lesions(rows, subjects, site.astype(object), study)

    is_skin = compartment.eq("skin")
    body_site = rows["region_location"].where(is_skin, rows["region_location"].fillna("faecal"))

    standard = pd.DataFrame({
        "sample": rows["sample_id"],
        "patient": subjects,
        "group": groups,
        "site": site,
        "severity": patient_stage,
        "skin_type": rows["region"].where(is_skin),
        "timepoint": rows["timepoint"] if "timepoint" in rows else np.nan,
        "pasi": patient_score,
        "body_site": body_site,
        "lesion_rank": lesion_rank,
        "lesion": lesion_label,
        "local_severity": local_severity_values(rows, site),
        "global_severity": visit_score,
    })

    return E.standardize_metadata(standard, study)


def load_studies(
    rank: str = "species",
    data_root: str | Path = DATA_ROOT,
    metadata_path: str | Path | None = None,
    source: str = "tsv",
    biom_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (relative-abundance matrix, standardized metadata, notes).

    source: 'tsv' (filtered Emu TSVs, as before), 'biom' (BIOM files below biom_dir, default
    <data_root>/BIOM_DIR) or 'both' (TSVs first; BIOM files add the samples without a TSV)."""
    source = (source or "tsv").strip().lower()
    if source not in SOURCES:
        raise ValueError(f"source must be one of {list(SOURCES)}, not {source!r}")
    root = Path(data_root)
    metadata_file = find_metadata(root, metadata_path)
    metadata_source = pd.read_csv(metadata_file, dtype=str)
    metadata_source.columns = [str(column).strip().lower() for column in metadata_source.columns]

    required = {
        "sample_id", "study", "subject", "group", "site", "lesion_status",
        "region_location", "region", "global_severity", "stage",
    }
    missing = required - set(metadata_source.columns)
    if missing:
        raise ValueError(
            f"{metadata_file} is missing required column(s): "
            f"{', '.join(sorted(missing))}"
        )
    known_samples = set(metadata_source["sample_id"].dropna().astype(str).str.strip())

    matrices = []
    metadata_frames = []
    notes = []

    biom = None
    if source != "tsv":
        bdir = resolve_biom_dir(root, biom_dir)
        biom = read_biom_folder(bdir)
        notes.append(f"Microbiome read from {SOURCES[source]}: {len(biom['files'])} BIOM file(s) with "
                     f"{len(biom['counts'])} sample(s) in {bdir}")
    biom_used: set[str] = set()
    wanted = {"tsv": "filtered TSV", "biom": "BIOM sample", "both": "filtered TSV or BIOM sample"}[source]

    for study, relative_dirs in STUDIES.items():
        results_dirs = [root / relative_dir for relative_dir in relative_dirs]
        md, study_notes = standardize_study_metadata(metadata_source, study)
        notes.extend(study_notes)
        mat = None
        if source in ("tsv", "both"):
            try:
                mat, ignored = load_filtered_rank(
                    results_dirs,
                    rank,
                    set(md["sample"]),
                    known_samples,
                )
                if ignored:
                    notes.append(
                        f"[{study}] Ignored {ignored} TSV(s) not present in the metadata"
                    )
            except FileNotFoundError as exc:
                if source == "tsv":
                    raise
                notes.append(f"[{study}] filtered TSVs not used: {exc}")

        if biom is not None:
            bmat, used, biom_notes = load_biom_rank(biom, rank, dict(zip(md["sample"], md["site"])), study)
            notes.extend(biom_notes)
            biom_used |= set(used.values())
            if mat is None or source == "biom":
                mat = bmat
            elif len(bmat):
                both = mat.index.intersection(bmat.index)
                extra = bmat.index.difference(mat.index)
                if len(both):
                    cols = mat.columns.union(bmat.columns)
                    diff = (mat.loc[both].reindex(columns=cols, fill_value=0.0)
                            - bmat.loc[both].reindex(columns=cols, fill_value=0.0)).abs().to_numpy().max()
                    notes.append(f"[{study}] {len(both)} sample(s) are in both the filtered TSVs and the BIOM "
                                 f"files: the TSV is used (largest difference in relative abundance: {diff:.4f})")
                if len(extra):
                    mat = pd.concat([mat, bmat.loc[extra]], axis=0).fillna(0.0)
                    notes.append(f"[{study}] {len(extra)} sample(s) without a filtered TSV were read from the "
                                 f"BIOM files")

        if mat is None or mat.empty:
            notes.append(f"[{study}] no {wanted} matched the metadata; study skipped")
            continue

        has_output = md["sample"].isin(mat.index)
        if (~has_output).any():
            missing_md = md.loc[~has_output]
            by_site = missing_md["site"].value_counts().to_dict()
            examples = missing_md["sample"].tolist()[:10]
            notes.append(
                f"[{study}] {int((~has_output).sum())} sample(s) had no matching {wanted} "
                f"(by site: {by_site}); examples: {examples}"
            )

        md = md.loc[has_output]
        mat = mat.loc[md["sample"].tolist()]
        mat.index = md.index

        matrices.append(mat)
        metadata_frames.append(md)

    if biom is not None:
        unused = sorted(set(biom["counts"]) - biom_used)
        if unused:
            ex = [f"{biom['info'][u]['sid']} ({Path(biom['info'][u]['file']).name})" for u in unused[:8]]
            notes.append(f"{len(unused)} BIOM sample(s) did not match a skin or gut sample_id of studies "
                         f"{', '.join(STUDIES)} in the metadata and were not used; examples: {ex}")
    if not matrices:
        msg = f"No {wanted} matched the metadata for any study"
        if biom is not None:
            meta_ids = metadata_source.loc[metadata_source["site"].fillna("").str.strip().str.lower().isin(COMPARTMENTS),
                                           "sample_id"].dropna().astype(str).tolist()[:5]
            ex = [f"{i['sid']} (source_file keys e.g. {sorted(k for k, v in biom['keys'].items() if u in v)[:3]})"
                  for u, i in list(biom["info"].items())[:3]]
            msg += (f". BIOM samples look like {ex}; metadata sample_ids look like {meta_ids}. A metadata sample_id "
                    f"must equal a BIOM sample ID or the Emu file name it came from.")
        raise FileNotFoundError(msg)

    combined_mat = pd.concat(matrices, axis=0).fillna(0.0)
    combined_mat = combined_mat.div(combined_mat.sum(axis=1), axis=0)
    combined_md = pd.concat(metadata_frames, axis=0)
    return combined_mat, combined_md, notes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and check filtered skin and gut microbiome data."
    )
    parser.add_argument("--rank", default="species")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--source", default="tsv", choices=list(SOURCES),
                        help="tsv (filtered Emu TSVs), biom (BIOM files) or both")
    parser.add_argument("--biom-dir", default=None,
                        help=f"folder with *.biom files (default: <data-root>/{BIOM_DIR})")
    args = parser.parse_args()

    mat, md, notes = load_studies(
        rank=args.rank,
        data_root=args.data_root,
        metadata_path=args.metadata,
        source=args.source,
        biom_dir=args.biom_dir,
    )

    print(f"\nLoaded {len(md)} samples and {mat.shape[1]} taxa.")
    print("\nSamples by study, group, and site:")
    print(md.groupby(["study", "group", "site"]).size().to_string())

    gut = md[md["site"] == "gut"]
    print(f"\nLoaded gut samples: {len(gut)}")
    if len(gut):
        print(gut.groupby(["study", "group", "severity"]).size().to_string())

    healthy = md[md["group"].str.lower() == "healthy"]
    print(f"\nLoaded healthy samples: {len(healthy)}")
    print(healthy.groupby(["study", "site"]).size().to_string())

    healthy_1806 = md[(md["study"] == "1806") & (md["group"] == "healthy")]
    print("Loaded healthy 1806 samples:", len(healthy_1806))
    print(healthy_1806["sample"].tolist())
    for note in notes:
        print(f"NOTE: {note}")


if __name__ == "__main__":
    main()
