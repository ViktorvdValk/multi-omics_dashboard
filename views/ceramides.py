"""Lipidomics page: ceramide classes, carbon number and per-ceramide analysis."""
from __future__ import annotations

import streamlit as st

import analysis as A
import skin_engine as E
from ceramide_loader import ceramide_summary
from views import _common as C, _omics


def _classes_tab(ds, md, mat, S):
    info = ds.feature_info
    levels = E.severity_levels(md)
    st.subheader("Class composition")
    by = st.radio("Group ceramides by", ["cer_class", "base_name", "fa_type"], horizontal=True, key="cer_by",
                  format_func={"cer_class": "class", "base_name": "sphingoid base", "fa_type": "fatty-acid type"}.get)
    C.plot(A.class_composition_figure(mat, md, info, levels, by=by), key="cer_classes")
    st.subheader("Carbon-number distribution")
    fig = A.chain_length_figure(mat, md, info, levels, E.severity_colors(levels))
    if fig is None:
        st.info("Carbon numbers could not be read from the ceramide names.")
    else:
        st.caption("Non-EO ceramides only; the carbon number is used as written in the name. Class fractions and "
                   "the mean carbon number are tested as 'Summary indices' in the statistical analysis. A ceramide "
                   "that was not detected counts as zero in these shares.")
        C.plot(fig, key="cer_chain")
    with st.expander("How the ceramide names were read"):
        C.table(ceramide_summary(ds))


def render():
    ds = C.get_dataset("ceramides", "Lipidomics (ceramides)")
    if ds is not None:
        _omics.render(ds, "ceramides", extra_tab=("Classes & carbon number", _classes_tab))
