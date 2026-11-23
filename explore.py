"""
explore.py -- select samples by criteria and look at any variable on them (logic of the Subset explorer page).

Everything is resolved onto the samples of ONE base dataset (for example the skin-microbiome samples):
  * a metadata variable is the sample's own value (site, timepoint, severity, local / global severity ...)
  * a feature or summary index of the base dataset is the sample's own value
  * a feature of ANOTHER dataset (or of another site of the base dataset) is the value of the same patient at
    the same timepoint at the chosen site (lesional: the patient's most severe lesion). Samples are never matched
    across timepoints; a sample without a counterpart has no value.
Criteria are comparisons on such variables; a sample without a value never meets a criterion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import skin_engine as E

META = "Metadata"
SAME = "same sample"
PATIENT_LOCAL = "local severity (patient's most severe lesion)"
META_VARS = ["severity", "group", "study", "site", "timepoint", "local_severity", "global_severity", "pasi",
             "skin_type", "body_site", "lesion", "patient"]
META_LABELS = {"local_severity": "local severity (of this lesion)", "global_severity": "global severity",
               "pasi": "PASI", "skin_type": "skin type", "body_site": "body site", "severity": "severity / stage",
               "group": "disease group"}
TRANSFORMS = {"compositional": {"rel": "relative abundance (%)", "clr": "CLR value", "present": "present (yes / no)"},
              "abundance": {"ln": "ln concentration", "conc": "concentration", "present": "measured (yes / no)"}}
NUMERIC_OPS = ["=", "≠", "≥", ">", "≤", "<", "between"]
TEXT_OPS = ["is one of", "is none of"]


@dataclass(frozen=True)
class Var:
    source: str                 # META or a dataset name ('skin', 'gut', 'ceramides', 'energy')
    name: str
    kind: str = "meta"          # meta | feature | index
    site: str = SAME            # SAME or a site of the source dataset
    transform: str = "value"

    def label(self, datasets) -> str:
        if self.source == META:
            return META_LABELS.get(self.name, self.name)
        ds = datasets[self.source]
        how = TRANSFORMS[ds.kind].get(self.transform, "") if self.kind == "feature" else ""
        where = "" if self.site == SAME else f", {self.site}"
        return f"{self.name}" + (f" · {how}" if how else "") + f" ({ds.label}{where})"


def meta_variables(base_md: pd.DataFrame, datasets: dict) -> list:
    out = [c for c in META_VARS if c in base_md and base_md[c].notna().any()
           and not (base_md[c].astype(str).str.strip() == "").all()]
    if any("local_severity" in ds.md and ds.md["local_severity"].notna().any() for ds in datasets.values()):
        out.insert(out.index("global_severity") if "global_severity" in out else len(out), PATIENT_LOCAL)
    return out


def feature_values(ds, name: str, kind: str, transform: str) -> pd.Series:
    """One value per sample of ds."""
    if kind == "index":
        return ds.derived[name]
    if transform == "rel":
        return 100 * ds.mat[name]
    if transform == "conc":
        return ds.mat[name].where(ds.mat[name] > 0)
    if transform == "present":
        yes, no = ("present", "absent") if ds.kind == "compositional" else ("measured", "not measured")
        return pd.Series(np.where(ds.mat[name] > 0, yes, no), index=ds.mat.index, dtype=object)
    return ds.values[name]                                   # clr | ln


def _key(md: pd.DataFrame) -> pd.Series:
    tp = md["timepoint"].astype(object).where(md["timepoint"].notna(), "")
    return md["patient"].astype(str) + "@" + tp.astype(str)


def _by_patient_timepoint(values: pd.Series, md: pd.DataFrame) -> pd.Series:
    """Mean of a numeric variable per patient x timepoint; for text the single value (None if they differ)."""
    k = _key(md)
    if pd.api.types.is_numeric_dtype(values):
        return values.groupby(k.values).mean()
    return values.groupby(k.values).agg(lambda s: s.dropna().iloc[0] if s.dropna().nunique() == 1 else None)


def patient_local_severity(datasets: dict) -> pd.Series:
    """Local severity of the most severe lesion per patient x timepoint (from any dataset that has it)."""
    parts = []
    for ds in datasets.values():
        md = ds.md[ds.md["site"].eq("lesional") & ds.md["local_severity"].notna()]
        if len(md):
            parts.append(pd.to_numeric(md["local_severity"], errors="coerce").groupby(_key(md).values).max())
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts).groupby(level=0).max()


def resolve(var: Var, base_name: str, base_md: pd.DataFrame, datasets: dict) -> pd.Series:
    """Values of `var` for the rows of base_md (index = base_md.index)."""
    if var.source == META:
        if var.name == PATIENT_LOCAL:
            s = pd.Series(patient_local_severity(datasets).reindex(_key(base_md).values).values, index=base_md.index)
        else:
            s = base_md[var.name]
            if var.name in ("local_severity", "global_severity", "pasi"):
                s = pd.to_numeric(s, errors="coerce")
            elif var.name == "timepoint":
                s = s.astype(object)
        return s.rename(var.name)
    ds = datasets[var.source]
    values = feature_values(ds, var.name, var.kind, var.transform)
    if var.source == base_name and var.site == SAME:
        return values.reindex(base_md.index).rename(var.name)
    src = E.most_severe_lesion(ds.md)
    src = (src.loc[E.stratum_samples(src, var.site)] if var.site in ("lesional", "non-lesional")
           else src[src["site"] == var.site])                # lesional / NL include the healthy reference
    lookup = _by_patient_timepoint(values.loc[src.index], src)
    return pd.Series(lookup.reindex(_key(base_md).values).values, index=base_md.index, name=var.name)


def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def text_levels(s: pd.Series, name: str = "") -> list:
    present = [v for v in pd.unique(s.dropna()) if str(v) != ""]
    if name == "severity":
        return E.severity_levels(pd.DataFrame({"severity": present}))
    if name == "timepoint":
        return E.timepoint_levels(pd.DataFrame({"timepoint": present}))
    order = {"absent": 0, "present": 1, "not measured": 0, "measured": 1}
    return sorted(map(str, present), key=lambda v: (order.get(v, 0), E._natural_key(v)))


def criterion_mask(s: pd.Series, op: str, value) -> pd.Series:
    """Rows meeting one criterion. A row without a value never meets it."""
    if op in TEXT_OPS:
        t = s.astype(object).map(lambda v: None if pd.isna(v) else str(v))
        hit = t.isin([str(v) for v in value])
        return (hit if op == "is one of" else ~hit) & t.notna()
    x = pd.to_numeric(s, errors="coerce")
    if op == "between":
        lo, hi = sorted(value)
        m = (x >= lo) & (x <= hi)
    else:
        v = float(value)
        tol = 1e-9 * max(1.0, abs(v))
        m = {"=": (x - v).abs() <= tol, "≠": (x - v).abs() > tol, "≥": x >= v - tol, ">": x > v + tol,
             "≤": x <= v + tol, "<": x < v - tol}[op]
    return m & x.notna()


def combine(masks: list, how: str, index) -> pd.Series:
    if not masks:
        return pd.Series(True, index=index)
    out = masks[0].copy()
    for m in masks[1:]:
        out = (out & m) if how == "all" else (out | m)
    return out


def describe(var_label: str, op: str, value) -> str:
    if op in TEXT_OPS:
        return f"{var_label} {op} {{{', '.join(map(str, value))}}}"
    if op == "between":
        return f"{var_label} between {value[0]:g} and {value[1]:g}"
    return f"{var_label} {op} {value:g}"
