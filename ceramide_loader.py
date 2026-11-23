"""
ceramide_loader.py -- ceramide (lipidomics) tables: name parsing, Dataset construction and summary indices.

Accepted names (edit parse_ceramide if yours differ):
    NS_C34, NdS_C42, EOS_C68, AH_C44      class + carbon number
    CER[NP] C44, Cer NS C24:0
    Cer(d18:1/24:0), Cer(t18:0/24:0(2OH)) sphingoid base / fatty acid

Class = fatty acid code (N non-hydroxy, A alpha-hydroxy, EO esterified omega-hydroxy, O omega-hydroxy)
      + sphingoid base code (dS dihydrosphingosine, S sphingosine, P phytosphingosine, H 6-hydroxysphingosine,
        SD sphingadiene, T dihydroxy-dihydrosphingosine).

The carbon number is used exactly as written in the name; no rule decides whether it is a total or a
fatty-acid chain length. It must mean the same thing for every ceramide of a file.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import datasets as DS

FEATURE_ALIASES = ("ceramide", "ceramides", "lipid", "lipid_species", "species", "analyte", "compound",
                   "feature", "variable", "parameter")
BASES = {"dS": "dihydrosphingosine", "S": "sphingosine", "P": "phytosphingosine", "H": "6-hydroxysphingosine",
         "SD": "sphingadiene", "T": "dihydroxy-dihydrosphingosine"}
FATTY_ACIDS = {"N": "non-hydroxy", "A": "alpha-hydroxy", "EO": "esterified omega-hydroxy", "O": "omega-hydroxy"}
_CLASS = re.compile(r"^(EO|N|A|O)(DS|SD|S|P|H|T)$", flags=re.I)


def _canonical_class(code):
    m = _CLASS.match(code.strip())
    if not m:
        return None
    fa, base = m.group(1).upper(), m.group(2).upper()
    return fa, {"DS": "dS"}.get(base, base)


def parse_ceramide(name) -> dict:
    out = dict(cer_class="unknown", fa_code="", base_code="", fa_type="unknown", base_name="unknown",
               carbons=np.nan, unsaturations=np.nan)
    s = re.sub(r"^\s*(cer(amide)?s?)\b[\s_\-:]*", "", str(name).strip(), flags=re.I)
    m = re.search(r"([dtm])(\d{2}):(\d)\s*/\s*(\d{2}):(\d)(.*)", s, flags=re.I)          # d18:1/24:0
    if m:
        prefix, base_unsat = m.group(1).lower(), int(m.group(3))
        hydroxy = bool(re.search(r"oh|;o|h\d", m.group(6), flags=re.I)) or bool(re.search(r"/\s*[ah]", s))
        base = {"d": "S" if base_unsat == 1 else "dS" if base_unsat == 0 else "SD",
                "t": "P" if base_unsat == 0 else "H", "m": "S"}[prefix]
        fa = "A" if hydroxy else "N"
        out.update(carbons=float(m.group(4)), unsaturations=float(m.group(5)))
    else:
        m = re.search(r"\[?\(?\s*([A-Za-z]{2,4})\s*\]?\)?[\s_\-]*C?\s*(\d{2,3})(?::(\d))?", s)
        cls = _canonical_class(m.group(1)) if m else None
        if not cls:
            return out
        fa, base = cls
        out.update(carbons=float(m.group(2)), unsaturations=float(m.group(3)) if m.group(3) else np.nan)
    out.update(cer_class=fa + base, fa_code=fa, base_code=base, fa_type=FATTY_ACIDS[fa], base_name=BASES[base])
    return out


def feature_info(names) -> pd.DataFrame:
    return pd.DataFrame([parse_ceramide(n) for n in names], index=pd.Index(names, name="ceramide"))


def summary_indices(mat: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """Per-sample indices: total (ln), class fractions (%), EO fraction, and the abundance-weighted mean carbon
    number of the non-EO ceramides. A ceramide that was not detected contributes zero."""
    info = info.reindex(mat.columns)
    total = mat.sum(axis=1)
    total = total.where(total > 0)
    out = pd.DataFrame({"Total ceramides (ln)": np.log(total)}, index=mat.index)
    for cls in sorted(c for c in info["cer_class"].unique() if c != "unknown"):
        out[f"% {cls}"] = 100 * mat.loc[:, (info["cer_class"] == cls).values].sum(axis=1) / total
    eo = (info["fa_code"] == "EO").values
    if eo.any():
        out["% EO classes"] = 100 * mat.loc[:, eo].sum(axis=1) / total
    sel = (~eo) & info["carbons"].notna().values
    if sel.sum() >= 2:
        sub, c = mat.loc[:, sel], info.loc[sel, "carbons"].values
        denom = sub.sum(axis=1)
        out["Mean carbon number, non-EO"] = (sub * c).sum(axis=1) / denom.where(denom > 0)
    return out


def ceramide_dataset(raw) -> DS.Dataset:
    """raw: one table, or a list of (table, study of its folder, file name) -- several files are combined; a
    table without a study column gets the study of its folder."""
    tables = raw if isinstance(raw, list) else [(raw, None, "")]
    parts, notes = [], []
    for df, study, label in tables:
        t, n = DS.tidy_table(df, FEATURE_ALIASES, "ceramides", default_study=study)
        parts.append(t)
        notes += [x.replace("[ceramides]", f"[ceramides] {label}:") if label and len(tables) > 1 else x for x in n]
    tidy = pd.concat(parts, ignore_index=True, sort=False)
    ds = DS.abundance_dataset("ceramides", "Ceramides", tidy, notes, "ceramide")
    ds.feature_info = feature_info(ds.mat.columns)
    unknown = int((ds.feature_info["cer_class"] == "unknown").sum())
    if unknown:
        ex = ds.feature_info.index[ds.feature_info["cer_class"] == "unknown"][:5].tolist()
        ds.notes.append(f"[ceramides] {unknown} name(s) could not be read as class + carbon number (e.g. {ex}); they "
                        f"are analysed as single ceramides but left out of the class and carbon-number indices")
    ds.derived = summary_indices(ds.mat, ds.feature_info)
    return ds


def ceramide_summary(ds: DS.Dataset) -> pd.DataFrame:
    info = ds.feature_info
    det = ds.mat.gt(0)
    return (pd.DataFrame({"class": info["cer_class"], "carbon number": info["carbons"],
                          "measured above zero in": det.mean(), "median (when measured)": ds.mat.where(det).median(),
                          "mean share of total (%)": 100 * ds.mat.div(ds.mat.sum(axis=1), axis=0).mean()})
            .rename_axis("ceramide").reset_index())
