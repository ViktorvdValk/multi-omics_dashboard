"""
skin_plots.py -- Plotly figure builders for the skin microbiome dashboard.

Visual encoding (everywhere):
  colour = severity   blue (mild) -> red (most severe), green = healthy
  shape  = site       circle = non-lesional, square = lesional, diamond = nostril,
                      triangle = gut (faecal), hexagon = pooled (patient mean of lesional + non-lesional)
"""
import itertools

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import skin_engine as E

SITE_SYMBOL = {"non-lesional": "circle", "lesional": "square", "healthy": "star", "nostril": "diamond",
               "gut": "triangle-up", "pooled": "hexagon", "L vs NL": "hexagon"}
SITE_ORDER = ["healthy", "non-lesional", "lesional", "nostril", "gut"]
STRATUM_ORDER = ["non-lesional", "lesional", "pooled", "healthy", "nostril", "gut"]
STRATUM_TITLE = {"pooled": "pooled (L + NL)", "non-lesional": "non-lesional", "lesional": "lesional",
                 "healthy": "healthy",
                 "nostril": "nostril", "gut": "gut (faecal)"}
EXTRA_SYMBOLS = ["cross", "x", "triangle-down", "pentagon", "bowtie", "hourglass", "star-square"]


def site_symbol(site):
    """Marker symbol for a site; specimen types unknown to the microbiome pages get a stable extra symbol."""
    if site not in SITE_SYMBOL:
        SITE_SYMBOL[site] = EXTRA_SYMBOLS[(len(SITE_SYMBOL) - 7) % len(EXTRA_SYMBOLS)]
    return SITE_SYMBOL[site]


def site_order(sites):
    present = set(sites)
    return [s for s in SITE_ORDER if s in present] + sorted(present - set(SITE_ORDER))


# continuous scale for local severity: blue (mild) -> red (most severe), like the severity colours
LOCAL_SCALE = [[0.0, "#2B6CB0"], [0.5, "#8E5BA8"], [1.0, "#C0392B"]]


def scale_colors(levels):
    """Evenly spaced colours of LOCAL_SCALE for ordered levels (local-severity groups)."""
    out = {}
    for i, lv in enumerate(levels):
        t = i / (len(levels) - 1) if len(levels) > 1 else 0.0
        out[lv] = px.colors.sample_colorscale(LOCAL_SCALE, [t])[0]
    return out


PALETTE = px.colors.qualitative.Set2 + px.colors.qualitative.Dark2
TEMPLATE = "plotly_white"
HEALTHY_COLOR = "#2E9E4F"            # green is reserved for healthy volunteers
NO_VALUE_COLOR = "#BBBBBB"
SCORE_LABELS = {"local_severity": "local severity", "global_severity": "global severity"}


# ------------------------------------------------------------------ gradient colouring (severity scores)
def gradient_spec(md, fill=None, line=None):
    """Colour points by a severity score instead of by stage: blue (lowest) -> red (highest), healthy = green,
    no value = grey. fill / line: 'local_severity' | 'global_severity' | None; with both, the marker fill shows
    one score and its outline the other. The range is taken from `md` so every plot of a page shares it.
    Returns None when nothing is to be coloured this way."""
    cols = [c for c in (fill, line) if c]
    if not cols:
        return None
    ranges = {}
    for c in cols:
        v = pd.to_numeric(md[c], errors="coerce") if c in md else pd.Series(dtype=float)
        if "group" in md and len(v):                        # healthy volunteers are green, not on the scale
            v = v[(md["group"] != E.HEALTHY).values]
        lo, hi = (float(v.min()), float(v.max())) if v.notna().any() else (0.0, 1.0)
        ranges[c] = (lo, hi if hi > lo else lo + 1.0)
    return {"fill": fill, "line": line, "range": ranges}


def gradient_color(value, lo, hi):
    """Colour of one value on LOCAL_SCALE (grey without a value)."""
    if pd.isna(value):
        return NO_VALUE_COLOR
    return px.colors.sample_colorscale(LOCAL_SCALE, [float(np.clip((value - lo) / (hi - lo), 0, 1))])[0]


def point_traces(x, y, meta, hover, symbol, sev_colors, grad=None, size=8.5, opacity=0.9):
    """Scatter traces for one group of points sharing a marker symbol. Without `grad` the points are coloured
    by meta['severity'] (sev_colors); with it by the score(s) of gradient_spec, healthy volunteers in green."""
    x, y, hover = np.asarray(x), np.asarray(y), np.asarray(hover, dtype=object)
    base = dict(size=size, symbol=symbol, opacity=opacity)
    traces = []
    if grad is None:
        sev = meta["severity"].astype(object).values
        for lv in pd.unique(sev):
            m = sev == lv
            traces.append(go.Scatter(x=x[m], y=y[m], mode="markers", text=hover[m], hoverinfo="text", showlegend=False,
                                     marker=dict(color=sev_colors.get(lv, "#888"), line=dict(width=0.8, color="#222"),
                                                 **base)))
        return traces
    healthy = (meta["group"].astype(object).values == E.HEALTHY) if "group" in meta else np.zeros(len(x), bool)
    num = lambda c: (pd.to_numeric(meta[c], errors="coerce").values.astype(float) if c and c in meta
                     else np.full(len(x), np.nan))
    fv, lv = num(grad["fill"]), num(grad["line"])
    fill_kind = np.where(healthy, "healthy", np.where(np.isnan(fv), "none", "value"))
    line_kind = np.where(healthy | np.isnan(lv), "none", "value")
    for fk in ("healthy", "value", "none"):
        for lk in ("value", "none"):
            m = (fill_kind == fk) & (line_kind == lk)
            if not m.any():
                continue
            marker = dict(base)
            if fk == "value":
                marker.update(color=fv[m], coloraxis="coloraxis")
            else:
                marker.update(color=HEALTHY_COLOR if fk == "healthy" else NO_VALUE_COLOR)
            marker["line"] = (dict(color=lv[m], coloraxis="coloraxis2", width=2.6) if lk == "value"
                              else dict(width=0.8, color="#222"))
            traces.append(go.Scatter(x=x[m], y=y[m], mode="markers", text=hover[m], hoverinfo="text",
                                     showlegend=False, marker=marker))
    return traces


def gradient_layout(fig, grad, shown_groups=(), row=None, col=None):
    """Colour bars (one per score) and legend entries 'healthy' / 'no value' for a gradient-coloured figure."""
    kw = dict(row=row, col=col) if row else {}
    if E.HEALTHY in set(shown_groups):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="healthy", legendgroup="grad",
                                 marker=dict(color=HEALTHY_COLOR, size=10, symbol="circle")), **kw)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="no value", legendgroup="grad",
                             marker=dict(color=NO_VALUE_COLOR, size=10, symbol="circle")), **kw)
    both = bool(grad["fill"] and grad["line"])
    for i, (key, axis) in enumerate([("fill", "coloraxis"), ("line", "coloraxis2")]):
        c = grad[key]
        if not c:
            continue
        lo, hi = grad["range"][c]
        title = (SCORE_LABELS.get(c, c).replace(" severity", "") + f"<br>({'fill' if key == 'fill' else 'outline'})"
                 if both else SCORE_LABELS.get(c, c))
        fig.update_layout(**{axis: dict(colorscale=LOCAL_SCALE, cmin=lo, cmax=hi,
                                        colorbar=dict(title=dict(text=title, font=dict(size=11)), len=0.5, y=0.25,
                                                      x=1.02 + 0.07 * i, thickness=14))})
    return fig


def gradient_caption(grad):
    parts = [f"{'fill' if k == 'fill' else 'outline'} = {SCORE_LABELS.get(grad[k], grad[k])}"
             for k in ("fill", "line") if grad[k]]
    if len(parts) == 1:
        parts = ["colour = " + parts[0].split(" = ")[1]]
    return " · ".join(parts)


def stars(q):
    return "***" if q < 0.001 else "**" if q < 0.01 else "*"


def categorical_colors(levels, severity_colors=None):
    if severity_colors and all(l in severity_colors for l in levels):
        return {l: severity_colors[l] for l in levels}
    return {l: PALETTE[i % len(PALETTE)] for i, l in enumerate(levels)}


def _rgba(color, alpha):
    if color.startswith("rgb("):
        rgb = color[4:-1].split(",")
        r, g, b = (int(value.strip()) for value in rgb)
    else:
        h = color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _hover(meta, y=None):
    timepoint = (
        meta["timepoint"].fillna("n/a").astype(str)
        if "timepoint" in meta
        else pd.Series("n/a", index=meta.index)
    )
    t = (meta["patient"].astype(str) + "<br>" + meta["study"].astype(str) + " · " + meta["group"].astype(str)
         + " · " + meta["severity"].astype(str) + " · timepoint: " + timepoint
         + "<br>skin type: " + meta["skin_type"].fillna("n/a").astype(str)
         + "<br>" + meta["body_site"].astype(str))
    for c, lab in SCORE_LABELS.items():
        if c in meta:
            t = t + pd.to_numeric(meta[c], errors="coerce").map(lambda v, lab=lab: "" if pd.isna(v) else f"<br>{lab}: {v:g}")
    if "lesion" in meta:
        t = t + meta["lesion"].astype(object).map(lambda v: f"<br>{v}" if isinstance(v, str) and v else "")
    if y is not None:
        t = t + "<br>value: " + pd.Series(np.asarray(y), index=meta.index).map(lambda v: f"{v:.4g}")
    return t


# ------------------------------------------------------------------ units
def build_units_long(mb, values, strata, factor):
    """One row per patient unit for every stratum x feature in `values`."""
    recs = []
    for st in strata:
        Y, meta = E.unit_frame(mb, values, st, factor=factor)
        if Y is None:
            continue
        for f in values.columns:
            d = meta.copy()
            d["row"], d["facet"], d["y"], d["x"] = f, st, Y[f].values, meta[factor].values
            d["hover"] = _hover(meta, Y[f].values)
            recs.append(d)
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()


def sig_items(tab, block, stratum, feature, x_levels, thr, qcol="q"):
    """[(i, j, stars)] for pairwise contrasts with q < thr."""
    if tab is None or not len(tab):
        return []
    sub = tab[(tab["block"] == block) & (tab["stratum"] == stratum) & (tab["feature"] == feature)
              & (tab["test"] == "pairwise") & (tab[qcol] < thr)]
    idx = {l: i for i, l in enumerate(x_levels)}
    return [(idx[r["lo"]], idx[r["hi"]], stars(r[qcol])) for r in sub.to_dict("records")
            if r["lo"] in idx and r["hi"] in idx]


# ------------------------------------------------------------------ box + points
def faceted_boxes(df, rows, facets, x_levels, sev_colors, sig=None, row_titles=None,
                  box_by_x=True, height_per_row=330, note=None, grad=None, facet_titles=None, point_colors=None):
    """Boxes + individual patient points. rows x facets grid.
    df columns: row, facet, x, y, severity, site, hover.  sig[(row, facet)] = [(i, j, label)].
    sev_colors: colour per x level (boxes). point_colors: colour per severity stage (points; default sev_colors).
    grad (see gradient_spec): points are coloured by a severity score (df needs group and the score columns)
    instead of by severity stage."""
    point_colors = sev_colors if point_colors is None else point_colors
    if not sig and len(df):                                 # leave out x levels without any point
        x_levels = [l for l in x_levels if l in set(df["x"])]
    R, C = len(rows), len(facets)
    facet_titles = facet_titles or {}
    titles = [facet_titles.get(f, STRATUM_TITLE.get(f, f)) if r == 0 else "" for r in range(R) for f in facets]
    fig = make_subplots(rows=R, cols=C, shared_yaxes="rows", subplot_titles=titles,
                        horizontal_spacing=0.04, vertical_spacing=min(0.12, 0.5 / max(R, 1)))
    rng = np.random.default_rng(0)
    sig = sig or {}
    for r, row in enumerate(rows):
        row_lo, row_hi = np.inf, -np.inf
        tops = {}
        for c, facet in enumerate(facets):
            sub = df[(df["row"] == row) & (df["facet"] == facet)] if len(df) else df
            if len(sub):
                lo, hi = sub["y"].min(), sub["y"].max()
                tops[c] = (lo, hi)
                row_lo, row_hi = min(row_lo, lo), max(row_hi, hi)
        span = (row_hi - row_lo) if np.isfinite(row_hi) and row_hi > row_lo else 1.0
        step = 0.11 * span
        ytop = row_hi
        for c, facet in enumerate(facets):
            sub = df[(df["row"] == row) & (df["facet"] == facet)] if len(df) else df
            for i, lv in enumerate(x_levels):
                v = sub[sub["x"] == lv] if len(sub) else sub
                if not len(v):
                    continue
                bc = sev_colors.get(lv, "#888888") if box_by_x else "#888888"
                fig.add_trace(go.Box(x=[i] * len(v), y=v["y"], width=0.55, line=dict(color=bc, width=1.5),
                                     fillcolor=_rgba(bc, 0.22), boxpoints=False, hoverinfo="skip",
                                     showlegend=False), row=r + 1, col=c + 1)
            for site, g in (sub.groupby("site") if len(sub) else []):
                xi = g["x"].map({l: i for i, l in enumerate(x_levels)}).astype(float).values
                xi = xi + rng.uniform(-0.17, 0.17, len(g))
                for tr in point_traces(xi, g["y"].values, g, g["hover"].values, site_symbol(site), point_colors, grad):
                    fig.add_trace(tr, row=r + 1, col=c + 1)
            for k, (i, j, lab) in enumerate(sorted(sig.get((row, facet), []), key=lambda t: (t[1] - t[0], t[0]))):
                y0 = ytop + step * (k + 1)
                h = step * 0.25
                for (xa, xb, ya, yb) in [(i, i, y0, y0 + h), (i, j, y0 + h, y0 + h), (j, j, y0 + h, y0)]:
                    fig.add_shape(type="line", x0=xa, x1=xb, y0=ya, y1=yb,
                                  line=dict(color="#222", width=1), row=r + 1, col=c + 1)
                fig.add_annotation(x=(i + j) / 2, y=y0 + h, text=lab, showarrow=False, yanchor="bottom",
                                   font=dict(size=13), row=r + 1, col=c + 1)
            n_top = max((len(sig.get((row, f), [])) for f in facets), default=0)
            counts = sub["x"].value_counts() if len(sub) else pd.Series(dtype=int)
            fig.update_xaxes(tickmode="array", tickvals=list(range(len(x_levels))),
                             ticktext=[f"{l} (n={int(counts.get(l, 0))})" for l in x_levels],
                             range=[-0.6, len(x_levels) - 0.4], tickfont=dict(size=10),
                             tickangle=-40 if len(x_levels) * C > 8 else 0,
                             row=r + 1, col=c + 1)
        n_top = max((len(sig.get((row, f), [])) for f in facets), default=0)
        pad = 0.06 * span
        for c in range(C):
            fig.update_yaxes(range=[row_lo - pad, ytop + step * (n_top + 1.2) + pad], row=r + 1, col=c + 1)
        if row_titles:
            fig.update_yaxes(title_text=row_titles[r], row=r + 1, col=1)
    seen = {s for s in df["severity"].unique()} if len(df) and grad is None else set()
    for sev in [l for l in point_colors if l in seen]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=sev, legendgroup="sev",
                                 marker=dict(color=point_colors[sev], size=10, symbol="circle")), row=1, col=1)
    seen_sites = {s for s in df["site"].unique()} if len(df) else set()
    for site in [s for s in list(SITE_SYMBOL) if s in seen_sites]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 name=(facet_titles or {}).get(site, STRATUM_TITLE.get(site, site)),
                                 legendgroup="site", marker=dict(color="#666", size=10,
                                                                 symbol=SITE_SYMBOL[site])), row=1, col=1)
    fig.update_layout(template=TEMPLATE, height=max(420, height_per_row * R), margin=dict(t=60, b=90),
                      legend=dict(title=f"{'colour = severity' if grad is None else gradient_caption(grad)}"
                                        f" · shape = site", orientation="v", y=1, yanchor="top"))
    if grad is not None:
        gradient_layout(fig, grad, df["group"].unique() if len(df) and "group" in df else (), row=1, col=1)
    if note:
        fig.add_annotation(text=note, xref="paper", yref="paper", x=0.5, y=-0.14 if R == 1 else -0.06,
                           showarrow=False, font=dict(size=10, color="#555"))
    return fig


# ------------------------------------------------------------------ PCoA
def pcoa_figure(coords, expl, md, sev_colors, color_by="severity", axes=(0, 1), title="", grad=None):
    """grad (see gradient_spec) colours the points by severity score(s) and overrides color_by."""
    a, b = axes
    if grad is not None:
        fig = go.Figure()
        for site in site_order(md["site"]):
            m = (md["site"] == site).values
            for tr in point_traces(coords[m, a], coords[m, b], md[m], _hover(md[m]), site_symbol(site), sev_colors,
                                   grad, size=10, opacity=0.88):
                fig.add_trace(tr)
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=STRATUM_TITLE.get(site, site),
                                     legendgroup="site", marker=dict(color="#666", size=10, symbol=site_symbol(site))))
        fig.update_layout(template=TEMPLATE, height=560, title=title,
                          xaxis_title=f"PCo{a + 1} ({expl[a]:.1%})", yaxis_title=f"PCo{b + 1} ({expl[b]:.1%})",
                          legend=dict(title=f"{gradient_caption(grad)} · shape = site", font=dict(size=10),
                                      y=1, yanchor="top"))
        return gradient_layout(fig, grad, md["group"].unique())
    if color_by != "severity" and pd.api.types.is_numeric_dtype(md[color_by]):     # gradient (local severity)
        cv = md[color_by].astype(float).values
        fig = go.Figure()
        for site in site_order(md["site"]):
            for has in (True, False):
                m = (md["site"] == site).values & (~np.isnan(cv) if has else np.isnan(cv))
                if m.any():
                    fig.add_trace(go.Scatter(
                        x=coords[m, a], y=coords[m, b], mode="markers", text=_hover(md[m]), hoverinfo="text",
                        name=site if has else f"{site} (no value)", legendgroup=site,
                        marker=dict(color=cv[m] if has else "#BBBBBB", coloraxis="coloraxis" if has else None,
                                    symbol=site_symbol(site), size=10, opacity=0.88, line=dict(width=0.9, color="#222"))))
        fig.update_layout(template=TEMPLATE, height=560, title=title,
                          xaxis_title=f"PCo{a + 1} ({expl[a]:.1%})", yaxis_title=f"PCo{b + 1} ({expl[b]:.1%})",
                          legend=dict(title="shape = site", font=dict(size=10)),
                          coloraxis=dict(colorscale=LOCAL_SCALE, colorbar=dict(title=color_by.replace("_", " "),
                                                                                len=0.6)))
        return fig
    levels = [l for l in (E.severity_levels(md) if color_by == "severity" else sorted(md[color_by].fillna("n/a").unique()))]
    col = categorical_colors(levels, sev_colors if color_by == "severity" else None)
    cvals = md[color_by].fillna("n/a") if color_by != "severity" else md["severity"]
    fig = go.Figure()
    for site in site_order(md["site"]):
        for lv in levels:
            m = ((md["site"] == site) & (cvals == lv)).values
            if not m.any():
                continue
            sub = md[m]
            fig.add_trace(go.Scatter(
                x=coords[m, a], y=coords[m, b], mode="markers", name=f"{lv} · {site}",
                text=_hover(sub), hoverinfo="text",
                marker=dict(color=col[lv], symbol=site_symbol(site), size=10, opacity=0.88,
                            line=dict(width=0.9, color="#222"))))
    fig.update_layout(template=TEMPLATE, height=560, title=title,
                      xaxis_title=f"PCo{a + 1} ({expl[a]:.1%})", yaxis_title=f"PCo{b + 1} ({expl[b]:.1%})",
                      legend=dict(title=f"{color_by} · site", font=dict(size=10)))
    return fig


# ------------------------------------------------------------------ composition
def composition_figure(matb, mdb, levels, top_n=12, split=None):
    top = matb.mean().nlargest(top_n).index
    df = matb[list(top)].copy()
    df["Other"] = (1 - df.sum(axis=1)).clip(lower=0)
    cols = ["severity", "site", "study", "skin_type", "group", "timepoint"]
    meta = mdb[cols + ([split] if split and split not in cols else [])].copy()
    meta["skin_type"] = meta["skin_type"].fillna("n/a")
    keys = ["severity", "site"] + ([split] if split else [])
    long = (df.join(meta).groupby(keys)[list(df.columns)].mean().reset_index()
            .melt(id_vars=keys, var_name="taxon", value_name="abundance"))
    cmap = {t: px.colors.qualitative.Alphabet[i % 26] for i, t in enumerate(top)}
    cmap["Other"] = "#BBBBBB"
    fig = px.bar(long, x="site", y="abundance", color="taxon", facet_col="severity",
                 facet_row=split, category_orders={"severity": [l for l in levels if l in set(long["severity"])],
                                                   "site": site_order(long["site"]), "taxon": list(top) + ["Other"]},
                 color_discrete_map=cmap, template=TEMPLATE,
                 height=420 if not split else 300 * long[split].nunique() + 120)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_yaxes(range=[0, 1], title_text="mean relative abundance", matches=None)
    fig.update_xaxes(title_text="")
    return fig


# ------------------------------------------------------------------ DA heatmap
def da_heatmap(da, block, thr, qcol="q", max_taxa=30, scale_label="log2 FC (CLR)"):
    pw = da[(da["block"] == block) & da["test"].isin(["pairwise", "association"])]
    if pw.empty:
        return None, "No pairwise comparisons available for this block."
    best = pw.groupby("feature")[qcol].min().sort_values()
    sel = best[best < 0.1].index.tolist()[:max_taxa]
    note = f"Taxa with {qcol} < 0.1 in at least one comparison (stars: {qcol} < 0.05 *, < 0.01 **, < 0.001 ***)."
    if not sel:
        sel = pw.groupby("feature")["p"].min().sort_values().index.tolist()[:15]
        note = f"No taxon reaches {qcol} < 0.1; showing the 15 with the lowest raw p (no stars)."
    order = {s: i for i, s in enumerate(STRATUM_ORDER)}
    cols = (pw[["stratum", "contrast"]].drop_duplicates()
            .assign(o=lambda d: d["stratum"].map(order)).sort_values("o", kind="stable"))
    cols = list(zip(cols["stratum"], cols["contrast"]))
    M = np.full((len(sel), len(cols)), np.nan)
    Q = np.full_like(M, np.nan)
    P = np.full_like(M, np.nan)
    for j, (s, c) in enumerate(cols):
        sub = pw[(pw["stratum"] == s) & (pw["contrast"] == c)].set_index("feature")
        for i, t in enumerate(sel):
            if t in sub.index:
                M[i, j], Q[i, j], P[i, j] = sub.at[t, "log2fc"], sub.at[t, qcol], sub.at[t, "p"]
    text = np.where(np.isnan(Q) | (Q >= thr), "", np.vectorize(stars)(np.nan_to_num(Q, nan=1)))
    xl = [f"{STRATUM_TITLE.get(s, s)}<br>{c}" for s, c in cols]
    hover = np.array([[f"{sel[i]}<br>{cols[j][0]} · {cols[j][1]}<br>log2FC {M[i, j]:.2f}<br>p {P[i, j]:.3g}"
                       f" · {qcol} {Q[i, j]:.3g}" if not np.isnan(M[i, j]) else ""
                       for j in range(len(cols))] for i in range(len(sel))])
    fig = go.Figure(go.Heatmap(z=M, x=xl, y=sel, zmin=-3, zmax=3, colorscale="RdBu_r", text=text,
                               texttemplate="%{text}", textfont=dict(size=14), hovertext=hover,
                               hoverinfo="text", colorbar=dict(title=f"{scale_label}<br>"
                                            + ("higher local severity" if (pw["test"] == "association").all()
                                               else "later vs earlier"))))
    fig.update_layout(template=TEMPLATE, height=max(380, 26 * len(sel) + 220), xaxis=dict(side="top", tickangle=-50),
                      yaxis=dict(autorange="reversed"), margin=dict(t=170))
    return fig, note


# ------------------------------------------------------------------ paired lines
def paired_figure(mb, matb, taxon, levels, sev_colors, pseudo=1e-4, timepoints=None, ylabel=None, grad=None):
    """Lesional vs non-lesional per patient: one column per severity, one row per selected timepoint.
    grad (see gradient_spec): a patient's line is coloured by the fill score (local severity = that of the
    lesional sample), healthy volunteers green."""
    score = (grad["fill"] or grad["line"]) if grad else None
    extra = [c for c in ("group", score) if c and c in mb]
    d = matb[[taxon]].join(mb[["patient", "site", "severity", "timepoint"] + extra])
    d = d[d["site"].isin(["lesional", "non-lesional"])]
    d["y"] = np.log10(d[taxon] + pseudo)
    both = d.groupby(["patient", d["timepoint"].astype(object).fillna("")])["site"].transform("nunique") == 2
    lv = [l for l in levels if l in set(d.loc[both, "severity"])]
    tps = [tp for tp in (timepoints or []) if tp in set(d.loc[both, "timepoint"].dropna())] or [None]
    if not lv:
        fig = go.Figure()
        fig.update_layout(template=TEMPLATE, height=200, xaxis=dict(visible=False), yaxis=dict(visible=False),
                          annotations=[dict(text="No patient has a lesional and a non-lesional sample at the same "
                                                 "timepoint in this selection.", showarrow=False)])
        return fig
    R, C = len(tps), max(len(lv), 1)
    fig = make_subplots(rows=R, cols=C, shared_yaxes=True, subplot_titles=lv + [""] * (R * C - len(lv)),
                        horizontal_spacing=min(0.03, 0.3 / C), vertical_spacing=min(0.12, 0.4 / R))
    for r, tp in enumerate(tps):
        for c, sev in enumerate(lv):
            subset = d[d["severity"] == sev]
            if tp is not None:
                subset = subset[subset["timepoint"] == tp]
            pv = subset.pivot_table(index="patient", columns="site", values="y", aggfunc="mean")
            pv = pv.reindex(columns=["non-lesional", "lesional"]).dropna()
            if score:
                sc = pd.to_numeric(subset[score], errors="coerce").groupby(subset["patient"]).max()
                hl = (subset.groupby("patient")["group"].first().eq(E.HEALTHY) if "group" in subset
                      else pd.Series(dtype=bool))
                lo, hi = grad["range"][score]
            for pat, row in pv.iterrows():
                colour = sev_colors.get(sev, "#888")
                if score:
                    colour = HEALTHY_COLOR if bool(hl.get(pat, False)) else gradient_color(sc.get(pat, np.nan), lo, hi)
                fig.add_trace(go.Scatter(x=["non-lesional", "lesional"], y=row.values, mode="lines+markers",
                                         line=dict(color=colour, width=1.4 if score else 1),
                                         opacity=0.8 if score else 0.55,
                                         marker=dict(size=6, symbol=["circle", "square"]), text=pat,
                                         hoverinfo="text+y", showlegend=False), row=r + 1, col=c + 1)
            if len(pv):
                fig.add_trace(go.Scatter(x=["non-lesional", "lesional"], y=pv.mean().values, mode="lines",
                                         line=dict(color="black", width=3.5), showlegend=False,
                                         hoverinfo="skip"), row=r + 1, col=c + 1)
            fig.update_xaxes(showticklabels=(r == R - 1), tickfont=dict(size=10), type="category",
                             categoryorder="array", categoryarray=["non-lesional", "lesional"],
                             ticktext=["NL", "L"], tickvals=["non-lesional", "lesional"], row=r + 1, col=c + 1)
        label = ylabel or f"log10(rel. abundance + {pseudo:g})"
        fig.update_yaxes(title_text=label if tp is None else f"{tp}<br>{label}", row=r + 1, col=1)
    fig.update_layout(template=TEMPLATE, height=max(420, 300 * R + 80), margin=dict(t=60))
    if score:                                              # colour bar for the line colours
        lo, hi = grad["range"][score]
        fig.add_trace(go.Scatter(x=["non-lesional"], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(color=[lo], colorscale=LOCAL_SCALE, cmin=lo, cmax=hi, showscale=True,
                                             colorbar=dict(title=SCORE_LABELS.get(score, score), len=0.8,
                                                           thickness=14))), row=1, col=1)
    return fig


# ------------------------------------------------------------------ points coloured by any variable (explorer)
def _colour_levels(colour, label):
    present = [v for v in pd.unique(colour.dropna())]
    if label == "severity":
        return E.severity_levels(pd.DataFrame({"severity": present})), None
    order = {"absent": 0, "present": 1, "not measured": 0, "measured": 1}
    levels = sorted(map(str, present), key=lambda v: (order.get(v, 0), E._natural_key(v)))
    return levels, None


def variable_traces(x, y, meta, colour=None, label="", size=10, opacity=0.88, extra_hover=None):
    """Points with shape = site, coloured by `colour` (a Series aligned with meta, or None for one neutral colour).
    A numeric variable is drawn on the blue -> red scale with a colour bar (healthy volunteers green when it is a
    severity score); a text variable gets one colour per level (severity: the stage colours). Points without a
    value are grey. Returns (traces, layout) -- pass layout to fig.update_layout."""
    x, y = np.asarray(x), np.asarray(y)
    hover = _hover(meta)
    if extra_hover is not None:
        hover = hover + "<br>" + pd.Series(np.asarray(extra_hover, dtype=object), index=meta.index).astype(str)
    sites = meta["site"].astype(object).values
    traces, layout = [], {}
    mk = lambda site, **kw: dict(size=size, opacity=opacity, symbol=site_symbol(site), line=dict(width=0.8, color="#222"), **kw)

    def add(mask, site, name=None, group=None, **marker):
        if mask.any():
            traces.append(go.Scatter(x=x[mask], y=y[mask], mode="markers", text=hover.values[mask], hoverinfo="text",
                                     showlegend=False, legendgroup=group, marker=mk(site, **marker)))

    def legend(name, color, group="colour"):
        traces.append(go.Scatter(x=[None], y=[None], mode="markers", name=name, legendgroup=group,
                                 marker=dict(color=color, size=10, symbol="circle")))

    if colour is None:
        for site in site_order(sites):
            add(sites == site, site, color="#5B7FA6")
    elif pd.api.types.is_numeric_dtype(colour) and not pd.api.types.is_bool_dtype(colour):
        cv = pd.to_numeric(colour, errors="coerce").values.astype(float)
        hover = hover + "<br>" + label + ": " + pd.Series(cv, index=meta.index).map(lambda v: "no value" if np.isnan(v) else f"{v:.4g}")
        is_score = colour.name in SCORE_LABELS or label in SCORE_LABELS.values()
        healthy = (meta["group"].astype(object).values == E.HEALTHY) if is_score else np.zeros(len(x), bool)
        on_scale = ~np.isnan(cv) & ~healthy
        for site in site_order(sites):
            s = sites == site
            add(s & on_scale, site, color=cv[s & on_scale], coloraxis="coloraxis")
            add(s & healthy, site, color=HEALTHY_COLOR)
            add(s & np.isnan(cv) & ~healthy, site, color=NO_VALUE_COLOR)
        if healthy.any():
            legend("healthy", HEALTHY_COLOR)
        if (np.isnan(cv) & ~healthy).any():
            legend("no value", NO_VALUE_COLOR)
        lo, hi = (np.nanmin(cv[on_scale]), np.nanmax(cv[on_scale])) if on_scale.any() else (0.0, 1.0)
        layout["coloraxis"] = dict(colorscale=LOCAL_SCALE, cmin=float(lo), cmax=float(hi if hi > lo else lo + 1),
                                   colorbar=dict(title=dict(text=label, font=dict(size=11)), len=0.55, y=0.28,
                                                 thickness=14))
    else:
        cs = colour.astype(object).where(colour.notna(), None)
        cs = cs.map(lambda v: None if v is None else str(v))
        levels, _ = _colour_levels(cs, label)
        cmap = (E.severity_colors(levels) if label == "severity" else categorical_colors(levels))
        if label != "severity" and E.HEALTHY in cmap:
            cmap[E.HEALTHY] = HEALTHY_COLOR
        hover = hover + "<br>" + label + ": " + cs.map(lambda v: "no value" if v is None else v)
        for site in site_order(sites):
            s = sites == site
            for lv in levels:
                add(s & (cs.values == lv), site, color=cmap[lv])
            add(s & cs.isna().values, site, color=NO_VALUE_COLOR)
        for lv in levels:
            legend(lv, cmap[lv])
        if cs.isna().any():
            legend("no value", NO_VALUE_COLOR)
    for site in site_order(sites):
        traces.append(go.Scatter(x=[None], y=[None], mode="markers", name=STRATUM_TITLE.get(site, site),
                                 legendgroup="site", marker=dict(color="#666", size=10, symbol=site_symbol(site))))
    layout["legend"] = dict(title=(f"colour = {label} · " if colour is not None else "") + "shape = site",
                            font=dict(size=10), y=1, yanchor="top")
    return traces, layout


def scatter_figure(x, y, meta, colour=None, label="", xlab="", ylab="", title="", height=560):
    fig = go.Figure()
    traces, layout = variable_traces(x, y, meta, colour, label)
    for t in traces:
        fig.add_trace(t)
    fig.update_layout(template=TEMPLATE, height=height, title=title, xaxis_title=xlab, yaxis_title=ylab, **layout)
    return fig


def category_figure(cat, y, meta, levels, colour=None, label="", xlab="", ylab="", height=520):
    """Box per level of a text variable with every sample as a point (coloured like scatter_figure)."""
    fig = go.Figure()
    cat = cat.astype(object)
    pos = {l: i for i, l in enumerate(levels)}
    for l in levels:
        v = y[(cat == l).values]
        if len(v):
            fig.add_trace(go.Box(x=[pos[l]] * len(v), y=v, width=0.55, line=dict(color="#888", width=1.5),
                                 fillcolor="rgba(136,136,136,0.15)", boxpoints=False, hoverinfo="skip", showlegend=False))
    xi = cat.map(pos).astype(float).values + np.random.default_rng(0).uniform(-0.17, 0.17, len(cat))
    traces, layout = variable_traces(xi, np.asarray(y), meta, colour, label, size=9)
    for t in traces:
        fig.add_trace(t)
    counts = cat.value_counts()
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(levels))), title_text=xlab,
                     ticktext=[f"{l}<br>n={int(counts.get(l, 0))}" for l in levels], range=[-0.6, len(levels) - 0.4])
    fig.update_layout(template=TEMPLATE, height=height, yaxis_title=ylab, **layout)
    return fig
