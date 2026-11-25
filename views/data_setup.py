"""Data setup: load any subset of the modalities, match patients and line up the timepoints."""
from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import datasets as DS
import discovery as D
import load_studies as L
from ceramide_loader import ceramide_dataset
from energy_loader import energy_dataset, split_by_metadata
from views import _common as C

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMEPOINT_MAP_FILE = os.path.join(APP_DIR, "timepoint_map.csv")     # your choices in the table are kept here

# Defaults (edit here, or set the environment variables before starting the app).
# Real data lives under /workspace/microbiome_analysis; defaults point there so the app
# loads out of the box. Override with the env vars (MULTIOMICS_*) if your paths differ.
WORKSPACE = "/workspace/microbiome_analysis"
DEFAULTS = {
    "cfg_root": os.environ.get("MULTIOMICS_DATA_ROOT", str(L.DATA_ROOT) if L.DATA_ROOT.is_dir() else ""),
    "cfg_rank": "species",
    "cfg_mb_source": os.environ.get("MULTIOMICS_MICROBIOME_SOURCE", "tsv"),    # tsv, biom or both
    "cfg_biom": os.environ.get("MULTIOMICS_BIOM_DIR", ""),                     # blank: <data folder>/biom
    "cfg_cer": os.environ.get("MULTIOMICS_CERAMIDES", f"{WORKSPACE}/ceramides/ceramides_all_long_v3.csv"),
    "cfg_blister": os.environ.get("MULTIOMICS_ENERGY_BLISTER",
                                  f"{WORKSPACE}/energy_metabolism/CHDR1806_SKAAL_METABOLOMICS_BLISTER_20261029_v1.xlsx"),
    "cfg_plasma": os.environ.get("MULTIOMICS_ENERGY_PLASMA",
                                 f"{WORKSPACE}/energy_metabolism/CHDR1806_SKAAL_METABOLOMICS_PLASMA_20261029_v1.xlsx"),
    # THE metadata file: the only source of group, severity, PASI, scores, skin type and body site (all modalities)
    "cfg_metaclin": os.environ.get("MULTIOMICS_METADATA") or os.environ.get(
        "MULTIOMICS_CLINICAL", f"{WORKSPACE}/meta_data_all/meta_data_all_complete.csv"),
}


@st.cache_data(show_spinner=False)
def _load_microbiome(root, rank, metadata, source="tsv", biom_dir=""):
    return L.load_studies(rank=rank, data_root=root, metadata_path=metadata or None,
                          source=source, biom_dir=biom_dir or None)


def read_table(source) -> pd.DataFrame:
    """Tab-, semicolon- or comma-separated text, or Excel, from a path or an uploaded file. Everything is read
    as text; numbers (decimal comma or point) are converted later."""
    name = (source if isinstance(source, str) else source.name).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(source, dtype=str)
    if isinstance(source, str):
        with open(source, "rb") as fh:
            data = fh.read()
    else:
        data = source.getvalue()
    text = data.decode("utf-8-sig", errors="replace")
    header = text.split("\n", 1)[0]
    sep = "\t" if "\t" in header else ";" if ";" in header else ","
    return pd.read_csv(io.StringIO(text), sep=sep, dtype=str)


def _saved_map():
    try:
        return pd.read_csv(TIMEPOINT_MAP_FILE, dtype=str)
    except (OSError, ValueError):
        return None


def _metadata_path(source, root, legacy=None):
    """The one metadata file: the given path or upload, else (legacy) the microbiome metadata argument, else
    meta_data_all_complete.csv (or the older names) in the sequencing folder. None if there is none."""
    if source is not None and not isinstance(source, str):            # an uploaded file -> a temporary file
        suffix = os.path.splitext(source.name)[1] or ".csv"
        with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as fh:
            fh.write(source.getvalue())
        return fh.name
    for given in (source, legacy):
        if given:
            return str(given)
    if root:
        try:
            return str(L.find_metadata(L.Path(root), None))
        except FileNotFoundError:
            return None
    return None


def load_bundle(root, metadata, rank, ceramide_source, blister_source, plasma_source, metaclin_source=None,
                saved_map=None, mb_source="tsv", biom_dir="") -> dict:
    """Each modality loads on its own: one failing or missing modality never blocks the others. All metadata
    (group, severity, PASI, global / local severity, skin type, body site) comes from ONE metadata file:
    metaclin_source, else `metadata`, else the one in the sequencing folder."""
    datasets, status, notes = {}, {}, []
    given = lambda s: s is not None and s != ""
    mb_source = mb_source if mb_source in L.SOURCES else "tsv"
    meta_file = _metadata_path(metaclin_source if given(metaclin_source) else None, root, metadata)
    ref = None
    if meta_file:
        try:
            ref = DS.load_metadata_reference(meta_file)
            notes += ref.notes
            status["Metadata file"] = f"loaded: {meta_file} ({len(ref.rows)} rows, {len(ref.patients)} patients)"
        except Exception as exc:
            status["Metadata file"] = f"not loaded: {exc}"
    else:
        status["Metadata file"] = "no metadata file given: group, severity and scores will be missing"
    if root or (mb_source != "tsv" and biom_dir):
        try:
            if not meta_file:
                raise FileNotFoundError("the microbiome needs the metadata file (sample ids, sites, timepoints)")
            mat, md, n = _load_microbiome(root, rank, meta_file, mb_source, biom_dir)
            datasets.update(DS.microbiome_datasets(mat, md, n))
            used = f" from {L.SOURCES[mb_source]}"
            status["Skin microbiome"] = ("loaded" + used) if "skin" in datasets else "no skin samples in the metadata"
            status["Gut microbiome"] = ("loaded" + used) if "gut" in datasets else "no gut samples in the metadata"
        except Exception as exc:                                    # show the reason, keep going
            status["Skin microbiome"] = status["Gut microbiome"] = f"not loaded: {exc}"
    else:
        status["Skin microbiome"] = status["Gut microbiome"] = "no sequencing folder given"
    if given(ceramide_source):
        try:
            datasets["ceramides"] = ceramide_dataset(read_table(ceramide_source))
            status["Ceramides"] = "loaded"
        except Exception as exc:
            status["Ceramides"] = f"not loaded: {exc}"
    else:
        status["Ceramides"] = "no file given"
    energy = {site: src for site, src in (("blister", blister_source), ("plasma", plasma_source)) if given(src)}
    if energy:
        try:
            datasets["energy"] = energy_dataset({site: read_table(src) for site, src in energy.items()})
            status["Energy metabolism"] = "loaded (" + " and ".join(energy) + ")"
        except Exception as exc:
            status["Energy metabolism"] = f"not loaded: {exc}"
    else:
        status["Energy metabolism"] = "no file given"
    tmap = None
    if datasets:
        notes += DS.harmonise(datasets, ref)
        tmap = DS.timepoint_map(datasets, saved_map, ref)
        notes += DS.apply_timepoint_map(datasets, tmap, ref)
    for ds in datasets.values():
        notes += ds.notes
    return {"datasets": datasets, "status": status, "notes": notes, "tmap": tmap, "rank": rank,
            "metadata_file": meta_file if ref is not None else None, "metadata_ref": ref}


def score_check(datasets: dict, metadata_file) -> pd.DataFrame:
    """Every sample of every dataset: how it was matched to the metadata file, the local and global severity
    written in the matched metadata row, and the value the app uses, so a wrong reading shows up at once."""
    m = DS.read_metadata_file(metadata_file)
    m.index = pd.MultiIndex.from_arrays([m["study"].map(DS.norm_study),
                                         m["sample_id"].astype(str).str.strip() if "sample_id" in m else ""])
    m = m[~m.index.duplicated()]
    rows = []
    for name, ds in datasets.items():
        md = ds.md
        match = (ds.metadata_match if ds.metadata_match is not None
                 else pd.DataFrame({"matched by": "no metadata file", "metadata sample": ""}, index=md.index))
        meta_sample = match["metadata sample"].reindex(md.index).fillna("").astype(str)
        file = m.reindex(pd.MultiIndex.from_arrays([md["study"].astype(str), meta_sample]))
        for c in ("local_severity", "global_severity"):
            raw = file[c] if c in file else pd.Series(np.nan, index=file.index)
            rows.append(pd.DataFrame({
                "dataset": ds.label, "study": md["study"].values, "sample": md["sample"].values,
                "patient": md["patient"].str.split("|").str[-1].values,
                "site (app)": md["site"].values, "timepoint (app)": md["timepoint"].astype(object).values,
                "matched by": match["matched by"].reindex(md.index).values, "metadata sample": meta_sample.values,
                "timepoint (file)": file["timepoint"].values if "timepoint" in file else np.nan,
                "lesion_status (file)": file["lesion_status"].values if "lesion_status" in file else np.nan,
                "score": c, "in file": raw.values, "used": pd.to_numeric(md[c], errors="coerce").values}))
    tab = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(tab):
        num = DS.to_number(tab["in file"])
        tab["differs"] = num.notna() & ~np.isclose(num.fillna(0), tab["used"].fillna(np.inf))
    return tab


def _score_check(bundle):
    path = bundle.get("metadata_file")
    datasets = bundle["datasets"]
    if not path or not datasets:
        return
    try:
        tab = score_check(datasets, path)
    except Exception as exc:                                  # a check must never block the page
        st.caption(f"Metadata check not available: {exc}")
        return
    n_diff = int(tab["differs"].sum()) if len(tab) else 0
    with st.expander(f"Check every sample against the metadata file ({n_diff} difference(s))", expanded=n_diff > 0):
        st.caption(f"File: {path}. 'matched by' says how the sample was matched to a metadata row: by sample id, or "
                   f"by patient + site + timepoint (+ lesion number); 'patient + timepoint' means only the visit was "
                   f"found (global severity, no local severity). 'in file' is the value written in the matched row, "
                   f"'used' the value in the plots and tests. Expected differences, not counted: a local severity is "
                   f"only kept for lesional samples; healthy volunteers have global severity 0; a row without its own "
                   f"global severity takes the value of the same patient at the same study and timepoint.")
        only = st.checkbox("Only rows that differ", value=n_diff > 0, key="chk_only")
        c1, c2 = st.columns(2)
        score = c1.radio("Score", ["local_severity", "global_severity"], horizontal=True, key="chk_score")
        which = c2.selectbox("Dataset", ["all"] + sorted(tab["dataset"].unique()), key="chk_dataset")
        show = tab[(tab["score"] == score) & ((tab["dataset"] == which) | (which == "all"))]
        C.table(show[show["differs"]] if only else show)


def _clear_results():
    for k in [k for k in st.session_state if str(k).endswith(("_analysis", "_results", "_result"))]:
        del st.session_state[k]


def _use_demo():
    import make_demo_data
    folder = st.session_state.get("demo_dir") or make_demo_data.main(tempfile.mkdtemp(prefix="multiomics_demo_"))
    st.session_state.update(demo_dir=folder, cfg_root=folder, cfg_rank="genus", cfg_mb_source="tsv",
                            cfg_cer=os.path.join(folder, "ceramides_long.csv"),
                            cfg_blister=os.path.join(folder, "energy_blister.txt"),
                            cfg_plasma=os.path.join(folder, "energy_plasma.txt"),
                            cfg_metaclin=os.path.join(folder, "meta_data_all3_new.csv"), _load_now=True,
                            cfg_data_mode="manual")


def render():
    st.title("Data setup")
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("cfg_drive", D.DEFAULT_DRIVE)
    st.session_state.setdefault("cfg_data_mode", "drive")
    if st.session_state["cfg_data_mode"] == "manual":
        st.button("Back to automatic loading from the data drive", on_click=_set_mode, args=("drive",))
        st.caption("Manual loading: give the paths (or upload the files) below and press Load data.")
        _render_manual()
    else:
        _render_drive()
        st.button("Manual loading (paths or uploads)", on_click=_set_mode, args=("manual",),
                  help="The V1 loader: one sequencing folder, one ceramide table, the energy files and one "
                       "metadata file, given by path or upload.")
    bundle = st.session_state.get("bundle")
    if bundle:
        _bundle_sections(bundle)


# =============================================================== V2: automatic loading from the data drive
# The combined metadata is a transient artifact — written to a temp dir so it never depends on
# (or leaves behind) a file in the app / repo directory. Available at deployment without any
# pre-existing file.
import tempfile
COMBINED_METADATA_FILE = os.path.join(tempfile.gettempdir(), "multiomics_combined_metadata.csv")


def _mtime(path) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


@st.cache_data(show_spinner="Scanning the data drive ...")
def _scan(drive, token):
    return D.scan(drive)


@st.cache_data(show_spinner=False)
def _read_microbiome(path, sheet, rank, mtime):
    if path.lower().endswith(".biom"):
        return L.read_biom_file(path, rank)
    return L.read_biotaviz(D.read_raw(path, sheet), rank, os.path.basename(path))


@st.cache_data(show_spinner=False)
def _read_full(path, sheet, mtime):
    return D.read_full(path, sheet)


@st.cache_data(show_spinner=False)
def _meta_values(path, mtime):
    return D.cell_values(path)


def _item_ids(item, rank) -> set:
    """The sample ids of a data file (lower case): Biotaviz / BIOM sample names, else its sample id column."""
    try:
        if item["modality"] == "microbiome":
            mat, _ = _read_microbiome(item["path"], item["sheet"], rank, _mtime(item["path"]))
            return {str(i).strip().lower() for i in mat.index}
        df = _read_full(item["path"], item["sheet"], _mtime(item["path"]))
    except Exception:
        return set()
    norm = [re.sub(r"[\s\-]+", "_", str(c).strip().lower()) for c in df.columns]
    col = next((df.columns[norm.index(c)] for c in DS.ID_COLUMNS if c in norm), None)
    if col is None:
        return set()
    return {v for v in df[col].fillna("").astype(str).str.strip().str.lower() if v and v != "nan"}


def match_metadata(chosen, studies, rank):
    return D.match_metadata(chosen, studies, lambda it: _item_ids(it, rank),
                            lambda p: _meta_values(p, _mtime(p)))


@st.cache_data(show_spinner=False)
def _combine(files):
    return D.combine_metadata([(p, study) for p, study, _ in files])


def _uid(item) -> str:
    return hashlib.md5(item["id"].encode()).hexdigest()[:10]


def _item_settings(item) -> dict:
    """The item with the modality / part chosen in the list (widgets keep their value across pages)."""
    u = _uid(item)
    mod = st.session_state.get(f"cfg_mod_{u}", item["modality"])
    parts = D.PARTS[mod]
    part = st.session_state.get(f"cfg_part_{u}", item["part"] if item["part"] in parts else parts[0])
    return {**item, "modality": mod, "part": part if part in parts else parts[0]}


def load_bundle_drive(items: list, metadata_files: list, rank: str, saved_map=None) -> dict:
    """V2: load the chosen data files of the drive with the COMBINED metadata of all study folders.
    items: dicts from discovery.scan (modality / part as chosen); metadata_files: [(path, study)]."""
    datasets, status, notes = {}, {}, []
    disc = pd.DataFrame()
    ref, meta_file, combined = None, None, None
    try:
        combined, disc, mnotes = _combine(tuple((str(p), s, _mtime(p)) for p, s in metadata_files))
        notes += mnotes
        # Recover the microbiome samples that are in the loaded data files but have no metadata row
        # (e.g. 2230 skin swabs only in the reference). Append their reference rows so the metadata
        # file and the loaded data agree, and only for studies actually being combined.
        ref_path = D._find_reference()
        if ref_path is not None:
            have = set(combined["sample_id"].astype(str).str.strip().str.lower())
            studies_in = {str(s) for s in combined["study"].dropna()}
            for it in items:
                if it.get("modality") != "microbiome":
                    continue
                try:
                    mat, _ = _read_microbiome(it["path"], it["sheet"], rank, _mtime(it["path"]))
                except Exception:
                    continue
                study = str(it.get("study") or "")
                if study not in studies_in:
                    continue
                missing = {str(x).strip().lower() for x in mat.index} - have
                if not missing:
                    continue
                rec, rnotes = L._recover_from_reference(missing, study, it.get("file", ""), ref_path)
                notes += rnotes
                if not rec:
                    continue
                base = len(combined)
                for row in rec:
                    combined.loc[base] = row
                    base += 1
                have |= {r["sample_id"].lower() for r in rec}
        combined.drop(columns=["_files"], errors="ignore").to_csv(COMBINED_METADATA_FILE, index=False)
        meta_file = COMBINED_METADATA_FILE
        ref = DS.load_metadata_reference(meta_file)
        notes += ref.notes
        status["Metadata"] = (f"{len(metadata_files)} file(s) combined ("
                              + ", ".join(os.path.basename(p) for p, _ in metadata_files)
                              + f"): {len(combined)} samples, {len(ref.patients)} patients")
    except Exception as exc:
        status["Metadata"] = f"not loaded: {exc}"
    by_mod = {}
    for it in items:
        by_mod.setdefault(it["modality"], []).append(it)
    if by_mod.get("microbiome"):
        try:
            if meta_file is None:
                raise FileNotFoundError("the microbiome needs the metadata (sample ids, sites, timepoints)")
            tables = []
            for it in by_mod["microbiome"]:
                mat, n = _read_microbiome(it["path"], it["sheet"], rank, _mtime(it["path"]))
                notes += [f"[microbiome] {x}" for x in n]
                comp = it["part"] if it["part"] in ("skin", "gut") else None      # checked against the metadata
                tables.append((mat, {"file": it["file"], "study": it["study"], "compartment": comp}))
            mat, md, n, flags, origin = L.assemble_microbiome(tables, meta_file)
            datasets.update(DS.microbiome_datasets(mat, md, n))
            if len(flags):
                disc = pd.concat([disc, flags], ignore_index=True)
            for key, label in (("skin", "Skin microbiome"), ("gut", "Gut microbiome")):
                if key not in datasets:
                    status[label] = f"no {key} samples in the loaded files"
                    continue
                ids = set(datasets[key].md["sample"].astype(str))
                used = sorted({origin[i] for i in ids if i in origin})
                status[label] = f"{len(ids)} samples from {len(used)} file(s): {', '.join(used)}"
        except Exception as exc:
            status["Skin microbiome"] = status["Gut microbiome"] = f"not loaded: {exc}"
    if by_mod.get("ceramides"):
        try:
            datasets["ceramides"] = ceramide_dataset(
                [(_read_full(it["path"], it["sheet"], _mtime(it["path"])), it["study"], it["file"])
                 for it in by_mod["ceramides"]])
            status["Ceramides"] = f"loaded from {len(by_mod['ceramides'])} file(s)"
        except Exception as exc:
            status["Ceramides"] = f"not loaded: {exc}"
    if by_mod.get("energy"):
        try:
            tables, enotes = {}, []
            for it in by_mod["energy"]:
                raw = _read_full(it["path"], it["sheet"], _mtime(it["path"]))
                split, n = split_by_metadata(raw, it["part"], it["study"], combined)   # site from 'sample source'
                enotes += n
                for part, df in split.items():
                    tables.setdefault(part, []).append((df, it["study"], it["file"]))
            datasets["energy"] = energy_dataset(tables, enotes)
            status["Energy metabolism"] = "loaded (" + ", ".join(f"{k}: {len(v)} file(s)" for k, v in tables.items()) + ")"
        except Exception as exc:
            status["Energy metabolism"] = f"not loaded: {exc}"
    tmap = None
    if datasets:
        notes += DS.harmonise(datasets, ref)
        tmap = DS.timepoint_map(datasets, saved_map, ref)
        notes += DS.apply_timepoint_map(datasets, tmap, ref)
    for ds in datasets.values():
        notes += ds.notes
    return {"datasets": datasets, "status": status, "notes": notes, "tmap": tmap, "rank": rank,
            "metadata_file": meta_file if ref is not None else None, "metadata_ref": ref,
            "discrepancies": disc, "source": "drive"}


def _toggle(ids, on):
    loaded = set(st.session_state.get("drive_loaded", []))
    loaded = loaded | set(ids) if on else loaded - set(ids)
    st.session_state["drive_loaded"] = sorted(loaded)


def _rescan():
    st.session_state["scan_token"] = st.session_state.get("scan_token", 0) + 1


def _render_drive():
    c1, c2, c3 = st.columns([3, 1.3, 1])
    c1.text_input("Data drive or folder", key="cfg_drive",
                  help="The folder that holds the study folders (CHDR1806_..., CHDR2230_...). Each has a 'metadata' and "
                       "a 'processed' folder. Default Z:\\ (or the environment variable MULTIOMICS_DRIVE).")
    c2.selectbox("Taxonomic rank", ["species", "genus", "family", "order", "class", "phylum"], key="cfg_rank")
    c3.write("")
    c3.button("Rescan", on_click=_rescan, help="Look for new or changed files.")
    studies, scan_notes = _scan(st.session_state["cfg_drive"].strip(), st.session_state.get("scan_token", 0))
    for n in scan_notes:
        st.warning(n)
    if not studies:
        return
    meta_files = [(str(p), stdy.study) for stdy in studies for p in stdy.metadata]
    items = [_item_settings(it) for stdy in studies for it in stdy.items]
    loaded = set(st.session_state.get("drive_loaded", []))
    chosen = [it for it in items if it["id"] in loaded and it["modality"] != "unknown"]
    rank = st.session_state["cfg_rank"]
    meta_used, meta_match, match_notes = match_metadata(chosen, studies, rank) if chosen else ([], pd.DataFrame(), [])
    used_paths = {p for p, _ in meta_used}
    st.caption(f"{len(studies)} study folder(s): " + ", ".join(f"{s.path.name}" for s in studies)
               + f". {len(meta_files)} metadata file(s), {len(items)} data file(s).")
    for stdy in studies:
        for n in stdy.notes:
            st.caption(f"⚠ {n}")
    with st.expander(f"Metadata files: {len(used_paths)} of {len(meta_files)} used for the loaded files", expanded=False):
        st.caption("Only the metadata of the loaded data files is combined: every metadata file of their own study "
                   "folders, plus a metadata file of another study folder that holds sample ids of a loaded file.")
        C.table(pd.DataFrame([{"used": "yes" if p in used_paths else "",
                               "study folder": Path(p).parent.parent.name if Path(p).parent.name.lower() == "metadata"
                               else Path(p).parent.name, "file": os.path.basename(p), "path": p}
                              for p, _ in meta_files]))
        if len(meta_match):
            st.markdown("**Metadata per loaded data file**")
            C.table(meta_match)
        for n in match_notes:
            st.caption(f"⚠ {n}")

    st.subheader("Data files")
    st.caption("The modality is recognised from the file's content; change it if it is wrong. Microbiome: skin or "
               "gut is taken from the metadata of every sample ('from metadata'); the file name is only checked "
               "against it. Nothing is loaded until you press Load.")
    recognised = [it["id"] for it in items if it["modality"] != "unknown"]
    b1, b2, _ = st.columns([1, 1, 4])
    b1.button("Load all", type="primary", on_click=_toggle, args=(recognised, True), disabled=not recognised)
    b2.button("Unload all", on_click=_toggle, args=([it["id"] for it in items], False), disabled=not loaded)
    widths = [0.7, 3.2, 1.9, 1.5, 2.4, 1]
    head = st.columns(widths)
    for col, title in zip(head, ["study", "file", "modality", "part", "recognised by", ""]):
        col.markdown(f"**{title}**")
    for it in items:
        u = _uid(it)
        row = st.columns(widths)
        row[0].write(it["study"])
        row[1].write(f"{it['modality folder'] + '/' if it['modality folder'] else ''}{it['file']}"
                     + (f"  \n*sheet {it['sheet']}*" if it["sheet"] else ""))
        st.session_state.setdefault(f"cfg_mod_{u}", it["modality"])
        row[2].selectbox("modality", list(D.MODALITIES), format_func=D.MODALITIES.get, key=f"cfg_mod_{u}",
                         label_visibility="collapsed")
        parts = D.PARTS[it["modality"]]
        if st.session_state.get(f"cfg_part_{u}") not in parts:
            st.session_state[f"cfg_part_{u}"] = it["part"] if it["part"] in parts else parts[0]
        row[3].selectbox("part", parts, key=f"cfg_part_{u}", label_visibility="collapsed", disabled=len(parts) == 1)
        row[4].caption(it["why"] or "-")
        on = it["id"] in loaded
        row[5].button("Unload" if on else "Load", key=f"btn_{u}", on_click=_toggle, args=([it["id"]], not on),
                      disabled=it["modality"] == "unknown" and not on, type="secondary" if on else "primary")

    signature = (tuple(sorted((it["id"], it["modality"], it["part"], _mtime(it["path"])) for it in chosen)),
                 tuple((p, _mtime(p)) for p, _ in meta_used), rank)
    bundle = st.session_state.get("bundle")
    if not chosen:
        if bundle is not None and bundle.get("source") == "drive":
            del st.session_state["bundle"]
            _clear_results()
        if os.path.exists(COMBINED_METADATA_FILE):          # only exists while drive files are loaded
            try:
                os.remove(COMBINED_METADATA_FILE)
            except OSError:
                pass
        return
    if bundle is None or bundle.get("source") != "drive" or bundle.get("signature") != signature:
        with st.spinner(f"Loading {len(chosen)} file(s) ..."):
            bundle = load_bundle_drive(chosen, meta_used, rank, _saved_map())
        bundle["notes"] = match_notes + bundle["notes"]
        bundle["signature"] = signature
        st.session_state["bundle"] = bundle
        _clear_results()


def _set_mode(mode):
    st.session_state["cfg_data_mode"] = mode


def _render_manual():
    types = ["csv", "tsv", "txt", "xlsx"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Microbiome (skin, gut)")
        st.text_input("Sequencing data folder", key="cfg_root")
        st.selectbox("Taxonomic rank", ["species", "genus", "family", "phylum"], key="cfg_rank")
        st.radio("Read the microbiome from", list(L.SOURCES), key="cfg_mb_source", format_func=L.SOURCES.get,
                 help="Filtered Emu TSVs: as before. BIOM files: the *.biom files of split_biom.py. Both: the TSVs "
                      "first, the BIOM files add samples without a TSV; for samples in both, the loading notes give "
                      "the largest difference, so you can check that the two inputs agree.")
        st.text_input("BIOM folder", key="cfg_biom", disabled=st.session_state.get("cfg_mb_source") == "tsv",
                      help=f"Blank: the '{L.BIOM_DIR}' folder in the sequencing data folder, or else the sequencing data "
                           "folder itself. Every *.biom file in it and its subfolders is read (a single .biom file "
                           "also works). Samples are matched to the metadata sample_id by BIOM sample ID or Emu file "
                           "name, within the study and site the file belongs to (e.g. emu_1806_gut.biom).")
    with c2:
        st.subheader("Ceramides")
        st.text_input("Ceramide table (path)", key="cfg_cer")
        cer_up = st.file_uploader("... or upload it", type=types, key="up_cer")
        st.caption("Long: study, subject, occ, location, ceramide, value. Or wide: one column per ceramide.")
    with c3:
        st.subheader("Energy metabolism (1806)")
        st.text_input("Blister-fluid file (path)", key="cfg_blister")
        bl_up = st.file_uploader("... or upload it", type=types, key="up_blister")
        st.text_input("Plasma file (path)", key="cfg_plasma")
        pl_up = st.file_uploader("... or upload it", type=types, key="up_plasma")
        st.caption("Columns: LCAB_ID, Subj_ID, Sampling_time, one column per metabolite. Decimal commas are fine.")
    # THE metadata file: sample ids, sites and timepoints of the microbiome, and all metadata of every dataset.
    with st.container():
        st.subheader("Metadata file")
        st.text_input("Metadata CSV (path)", key="cfg_metaclin",
                      help="Blank: meta_data_all_complete.csv in the sequencing data folder.")
        mc_up = st.file_uploader("... or upload it", type=types, key="up_metaclin")
        st.caption("The only source of group, stage / severity, PASI, global and local severity, skin type and body "
                   "site, for every dataset. Microbiome samples are its rows (sample_id); ceramide and energy samples "
                   "are matched to its rows by sample id, or by patient, site, timepoint and lesion number.")
    b1, b2 = st.columns([1, 3])
    load = b1.button("Load data", type="primary")
    b2.button("Try with synthetic demo data", on_click=_use_demo)
    if load or st.session_state.pop("_load_now", False):
        pick = lambda up, key: up if up is not None else st.session_state[key].strip()
        with st.spinner("Loading ..."):
            st.session_state["bundle"] = load_bundle(
                st.session_state["cfg_root"].strip(), "", st.session_state["cfg_rank"],
                pick(cer_up, "cfg_cer"), pick(bl_up, "cfg_blister"), pick(pl_up, "cfg_plasma"),
                pick(mc_up, "cfg_metaclin"),
                None if st.session_state.get("demo_dir") == st.session_state["cfg_root"].strip() else _saved_map(),
                mb_source=st.session_state["cfg_mb_source"], biom_dir=st.session_state["cfg_biom"].strip())
        _clear_results()


def _bundle_sections(bundle):
    st.subheader("Status")
    C.table(pd.DataFrame([{"modality": k, "status": v} for k, v in bundle["status"].items()]))
    disc = bundle.get("discrepancies")
    if disc is not None and bundle.get("source") == "drive":
        with st.expander(f"Metadata discrepancies ({len(disc)})", expanded=len(disc) > 0):
            if not len(disc):
                st.write("The metadata files agree with each other and with the data files.")
            else:
                st.caption("Where metadata files disagree, the file in the study's own CHDR folder wins (if its files "
                           "agree among themselves); otherwise the value is left missing. 'used' is the value the "
                           "dashboard uses; 'values' every value found, with its file.")
                c1, c2 = st.columns(2)
                kinds = c1.multiselect("Kind", sorted(disc["kind"].unique()), key="disc_kind")
                fields = c2.multiselect("Field", sorted(disc["field"].unique()), key="disc_field")
                show = disc[(disc["kind"].isin(kinds) | (not kinds)) & (disc["field"].isin(fields) | (not fields))]
                C.table(show)
                st.download_button("Download the list (CSV)", disc.to_csv(index=False), "metadata_discrepancies.csv",
                                   key="disc_dl")
        if bundle.get("metadata_file") and os.path.isfile(bundle["metadata_file"]):
            with open(bundle["metadata_file"], "rb") as fh:
                st.download_button("Download the combined metadata (CSV)", fh.read(), "combined_metadata.csv",
                                   key="meta_dl")
    datasets = bundle["datasets"]
    if not datasets:
        return

    st.subheader("Timepoints")
    st.caption("Samples of different modalities are matched on study, patient and timepoint. Every file has its "
               "own labels, so each label is assigned to baseline, mid or follow-up here. The proposal takes the "
               "earliest label of a study as baseline and the latest as follow-up: check it, especially where a "
               "file has only two of three visits. A blank or 'nvt' label counts as baseline.")
    edited = st.data_editor(
        bundle["tmap"], hide_index=True, key="tp_editor", disabled=["dataset", "study", "label in file", "samples"],
        column_config={"timepoint": st.column_config.SelectboxColumn("timepoint", options=DS.CANONICAL_TIMEPOINTS,
                                                                     required=True)}, **C.STRETCH_DF)
    if st.button("Apply timepoint mapping", disabled=edited.equals(bundle["tmap"])):
        bundle["tmap"] = edited.reset_index(drop=True)
        bundle["tp_notes"] = DS.apply_timepoint_map(datasets, bundle["tmap"], bundle.get("metadata_ref"))
        if st.session_state.get("demo_dir") != st.session_state["cfg_root"].strip():
            try:
                bundle["tmap"].to_csv(TIMEPOINT_MAP_FILE, index=False)
            except OSError:
                pass
        _clear_results()
        st.rerun()

    st.subheader("Patients per study, timepoint and modality")
    C.table(DS.coverage(datasets))
    _score_check(bundle)
    notes = bundle["notes"] + bundle.get("tp_notes", [])
    if notes:
        with st.expander(f"Loading notes ({len(notes)}): patient matching, missing values", expanded=True):
            for n in notes:
                st.write("•", n)
