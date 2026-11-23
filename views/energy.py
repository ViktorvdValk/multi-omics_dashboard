"""Energy-metabolism page: succinate, fumarate, malate, isocitrate in lesional and non-lesional blister fluid and
in plasma, plus ratios."""
from __future__ import annotations

import streamlit as st

import skin_engine as E
from energy_loader import add_ratio, refresh_ratios
from views import _common as C, _omics


def _ratios_tab(ds, md, mat, S):
    st.caption("A ratio is added as a summary index: it gets its own box plot (Features tab), is tested in the "
               "statistical analysis and can colour ordinations on the Integrated page.")
    feats = list(ds.mat.columns)
    c1, c2, c3 = st.columns([2, 2, 1])
    num = c1.selectbox("Numerator", feats, key="energy_num")
    den = c2.selectbox("Denominator", feats, index=min(1, len(feats) - 1), key="energy_den")
    if c3.button("Add ratio", key="energy_addratio") and num != den:
        st.success(f"Added {add_ratio(ds, num, den)}")
    if ds.derived is None or ds.derived.empty:
        st.info("No ratios defined yet, for example succinate / fumarate.")
        return
    ratio = st.selectbox("Show", list(ds.derived.columns), key="energy_ratio")
    levels, strata = E.severity_levels(md), E.strata_for(set(md["site"]))
    grad = C.plot_grad(md, "energy")
    fig = C.unit_boxes(md, ds.derived.loc[md.index, [ratio]], strata, "severity", levels, E.severity_colors(levels),
                       E.severity_colors(levels), grad=grad, average=S.timepoint_mode == "average",
                       site_titles=ds.site_titles or None)
    if fig is not None:
        C.plot(fig, key="energy_ratio_box")


def render():
    ds = C.get_dataset("energy", "Energy metabolism")
    if ds is not None:
        _omics.render(ds, "energy", extra_tab=("Ratios", _ratios_tab), prepare=refresh_ratios)
