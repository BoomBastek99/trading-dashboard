"""
Correlation Break Detector.

Monitors a set of asset pairs for departures from their historical correlation.
A z-score derived from the 60-day rolling correlation against its own long-run
mean/std produces a four-tier severity (Normal / Notable / Significant /
Extreme), with historical context: when this pair broke this hard before, what
did the two assets do over the next 5/10/20 trading days?

Run with:
    py -3.13 -m streamlit run correlation_break_detector.py --server.port=8509
"""

from __future__ import annotations

import json
import os
import textwrap
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_provider import clear_cache as _clear_data_cache, get_history
from design_system import *  # noqa: F401,F403

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Correlation Break Detector",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


def render_html(s: str) -> None:
    st.markdown(textwrap.dedent(s).strip(), unsafe_allow_html=True)


# Inject pulse animation for Extreme cards
render_html("""
    <style>
    @keyframes corrbreak-pulse {
      0%, 100% { box-shadow: 0 0 14px rgba(255,23,68,0.55), 0 0 28px rgba(255,23,68,0.30); }
      50%      { box-shadow: 0 0 32px rgba(255,23,68,0.95), 0 0 56px rgba(255,23,68,0.65); }
    }
    .corrbreak-extreme {
      animation: corrbreak-pulse 1.6s ease-in-out infinite;
    }
    </style>
""")


# ── Defaults ───────────────────────────────────────────────────────────────

DEFAULT_PAIRS = [
    ("SPY", "QQQ"),
    ("GLD", "TLT"),
    ("SPY", "IWM"),
    ("BTC-USD", "ETH-USD"),
    ("SPY", "EEM"),
]

ALERT_FILE = "correlation_alerts.json"


# ── Data ───────────────────────────────────────────────────────────────────

def fetch_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Thin wrapper around the unified data provider (FMP → TwelveData → yfinance)."""
    return get_history(ticker, start, end)


# ── Pair analysis ──────────────────────────────────────────────────────────

@dataclass
class PairAnalysis:
    pair: tuple[str, str]
    rolling_60: pd.Series       # 60-day rolling correlation
    rolling_20: pd.Series       # 20-day rolling correlation
    returns_a: pd.Series        # daily returns for asset A
    returns_b: pd.Series        # daily returns for asset B
    mean: float                 # historical mean of 60-day rolling corr
    std: float                  # historical std of 60-day rolling corr
    current_corr: float
    current_z: float
    severity: str
    severity_color: str


SEVERITY_ORDER = {"Normal": 0, "Notable": 1, "Significant": 2, "Extreme": 3}


def classify(z: float) -> tuple[str, str]:
    if z <= -2.5:
        return "Extreme", ACCENT_RED
    if z <= -2.0:
        return "Significant", "#ff6b1a"  # orange
    if z <= -1.5:
        return "Notable", ACCENT_AMBER
    return "Normal", TEXT_MUTED


def analyze_pair(pair: tuple[str, str], data: dict[str, pd.DataFrame]) -> PairAnalysis | None:
    a, b = pair
    if a not in data or b not in data or data[a].empty or data[b].empty:
        return None
    common = data[a].index.intersection(data[b].index)
    if len(common) < 80:
        return None
    closes_a = data[a]["Close"].reindex(common).astype(float)
    closes_b = data[b]["Close"].reindex(common).astype(float)
    ra = closes_a.pct_change()
    rb = closes_b.pct_change()

    r60 = ra.rolling(60).corr(rb).dropna()
    r20 = ra.rolling(20).corr(rb).dropna()
    if len(r60) < 30:
        return None

    mean = float(r60.mean())
    std = float(r60.std())
    cur = float(r60.iloc[-1])
    z = float((cur - mean) / std) if std > 0 else 0.0
    severity, color = classify(z)
    return PairAnalysis(
        pair=pair,
        rolling_60=r60, rolling_20=r20,
        returns_a=ra.dropna(), returns_b=rb.dropna(),
        mean=mean, std=std,
        current_corr=cur, current_z=z,
        severity=severity, severity_color=color,
    )


# ── Historical context (forward returns after past breaks) ─────────────────

def cluster_dates(dates: list[pd.Timestamp], min_gap_days: int = 30) -> list[pd.Timestamp]:
    """Keep only the first date in each contiguous-ish cluster."""
    if not dates:
        return []
    sorted_dates = sorted(dates)
    out = [sorted_dates[0]]
    for d in sorted_dates[1:]:
        if (d - out[-1]).days >= min_gap_days:
            out.append(d)
    return out


def historical_breaks(pa: PairAnalysis, severity_threshold: float = -2.0) -> pd.DataFrame:
    z_series = (pa.rolling_60 - pa.mean) / (pa.std + 1e-12)
    break_dates = z_series[z_series <= severity_threshold].index.tolist()
    # Drop the most recent break if it's the same as 'current' to avoid self-counting
    today = pa.rolling_60.index[-1]
    break_dates = [d for d in break_dates if (today - d).days >= 30]
    if not break_dates:
        return pd.DataFrame()

    clustered = cluster_dates(break_dates, min_gap_days=30)

    a, b = pa.pair
    rows = []
    for d in clustered:
        try:
            i_a = pa.returns_a.index.get_loc(d)
            i_b = pa.returns_b.index.get_loc(d)
        except KeyError:
            continue
        record = {"date": d.date(), "z": float(z_series.loc[d])}
        for h in [5, 10, 20]:
            ea = i_a + h
            eb = i_b + h
            if ea < len(pa.returns_a) and eb < len(pa.returns_b):
                fwd_a = (1.0 + pa.returns_a.iloc[i_a + 1:ea + 1]).prod() - 1.0
                fwd_b = (1.0 + pa.returns_b.iloc[i_b + 1:eb + 1]).prod() - 1.0
            else:
                fwd_a = float("nan")
                fwd_b = float("nan")
            record[f"{a}+{h}d"] = fwd_a * 100
            record[f"{b}+{h}d"] = fwd_b * 100
        rows.append(record)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)
    return df


# ── Alert log ──────────────────────────────────────────────────────────────

def load_alerts() -> list[dict]:
    if not os.path.exists(ALERT_FILE):
        return []
    try:
        with open(ALERT_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def append_alerts(new: list[dict], cap: int = 200) -> None:
    existing = load_alerts()
    existing.extend(new)
    existing = existing[-cap:]
    try:
        with open(ALERT_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, default=str)
    except Exception:
        pass


# ── Sidebar ────────────────────────────────────────────────────────────────

def parse_pairs(s: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in s.split(","):
        chunk = chunk.strip()
        if "/" not in chunk:
            continue
        a, _, b = chunk.partition("/")
        a, b = a.strip().upper(), b.strip().upper()
        if not a or not b or (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((a, b))
    return out


with st.sidebar:
    st.markdown(section_header("Pairs"), unsafe_allow_html=True)
    default_pairs_str = ", ".join(f"{a}/{b}" for a, b in DEFAULT_PAIRS)
    pairs_str = st.text_area("Format: A/B, C/D, ...", value=default_pairs_str, height=110)
    pairs = parse_pairs(pairs_str)

    st.markdown(section_header("Settings"), unsafe_allow_html=True)
    today = pd.Timestamp.today().normalize()
    default_start = (today - pd.DateOffset(years=3)).date()
    date_range = st.date_input(
        "Date range",
        value=(default_start, today.date()),
        max_value=today.date(),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d, end_d = default_start, today.date()

    z_threshold = st.slider("Alert z-score threshold", -3.5, -0.5, -2.0, step=0.1)

    run = st.button("Check Correlations", type="primary", use_container_width=True)


# ── Run ────────────────────────────────────────────────────────────────────

if "cb_results" not in st.session_state:
    st.session_state.cb_results = None

if not pairs:
    render_html(f"<div style='color:{TEXT_MUTED};padding:64px 0;text-align:center;'>"
                f"Add at least one pair like <code>SPY/QQQ</code> in the sidebar.</div>")
    st.stop()

if run or st.session_state.cb_results is None:
    if run:
        _clear_data_cache()
    with st.spinner("Downloading prices and computing correlations..."):
        unique_tickers = sorted(set(t for p in pairs for t in p))
        progress = st.progress(0.0, text="Starting...")
        data: dict[str, pd.DataFrame] = {}
        for i, t in enumerate(unique_tickers):
            progress.progress(i / len(unique_tickers), text=f"Fetching {t}...  ({i+1}/{len(unique_tickers)})")
            try:
                data[t] = fetch_history(t, str(start_d), str(end_d))
            except Exception:
                data[t] = pd.DataFrame()
        progress.empty()

        analyses: list[PairAnalysis] = []
        skipped: list[tuple[tuple[str, str], str]] = []
        for p in pairs:
            pa = analyze_pair(p, data)
            if pa is None:
                missing = [t for t in p if t not in data or data[t].empty]
                reason = f"missing data for {','.join(missing)}" if missing else "insufficient overlap"
                skipped.append((p, reason))
            else:
                analyses.append(pa)

        # Append to alert log for any breaks
        new_alerts = []
        ts = datetime.now(timezone.utc).isoformat()
        for pa in analyses:
            if pa.severity != "Normal":
                new_alerts.append({
                    "timestamp": ts,
                    "pair": f"{pa.pair[0]}/{pa.pair[1]}",
                    "current_corr": round(pa.current_corr, 4),
                    "z_score": round(pa.current_z, 3),
                    "severity": pa.severity,
                })
        if new_alerts:
            append_alerts(new_alerts)

        st.session_state.cb_results = {"analyses": analyses, "skipped": skipped, "z_threshold": z_threshold}


R = st.session_state.cb_results
analyses: list[PairAnalysis] = R["analyses"]
skipped = R["skipped"]
z_threshold_for_history = R["z_threshold"]

# Source banner
n_breaks = sum(1 for pa in analyses if pa.severity != "Normal")
render_html(f"""
    <div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;color:{TEXT_MUTED};
                letter-spacing:2px;margin-bottom:14px;'>
      {status_dot('connected')}{len(analyses)} PAIRS MONITORED &nbsp;·&nbsp;
      {n_breaks} BREAK{'S' if n_breaks != 1 else ''} ACTIVE &nbsp;·&nbsp;
      THRESHOLD z &lt;= {z_threshold_for_history:.1f} &nbsp;·&nbsp;
      WINDOW 60d
    </div>
""")

if skipped:
    bullets = "".join(f"<li>{a}/{b}: {why}</li>" for (a, b), why in skipped)
    render_html(f"""
        <div style='background:rgba(255,193,7,0.08);border:1px solid {ACCENT_AMBER};
                    border-left:3px solid {ACCENT_AMBER};border-radius:10px;
                    padding:10px 16px;margin-bottom:14px;
                    font-family:DM Sans;font-size:0.78rem;color:{TEXT_SECONDARY};'>
          <b style='color:{ACCENT_AMBER};text-transform:uppercase;letter-spacing:2px;font-size:0.7rem;'>
            Skipped pairs</b>
          <ul style='margin:6px 0 0 0;padding-left:18px;'>{bullets}</ul>
        </div>
    """)

if not analyses:
    render_html(f"<div style='color:{TEXT_MUTED};padding:48px;text-align:center;'>"
                f"No analyses available. yfinance may be rate-limited; wait, then "
                f"click <b style='color:{ACCENT_CYAN}'>Check Correlations</b>.</div>")
    st.stop()


# ── Status cards row ───────────────────────────────────────────────────────

cards = st.columns(len(analyses))
for i, pa in enumerate(analyses):
    severity = pa.severity
    color = pa.severity_color
    pair_label = f"{pa.pair[0]} / {pa.pair[1]}"

    if severity == "Normal":
        pill_bg = "rgba(255,255,255,0.04)"
        pill_color = TEXT_MUTED
        card_border = BORDER
        card_extra_class = ""
        card_extra_style = ""
    else:
        pill_bg = f"{color}33"  # ~20% alpha
        pill_color = color
        card_border = color
        if severity == "Extreme":
            card_extra_class = "corrbreak-extreme"
            card_extra_style = ""
        elif severity == "Significant":
            card_extra_class = ""
            card_extra_style = f"box-shadow: 0 0 22px {color}55;"
        else:  # Notable
            card_extra_class = ""
            card_extra_style = f"box-shadow: 0 0 14px {color}33;"

    with cards[i]:
        render_html(f"""
            <div class='{card_extra_class}' style='background:{BG_CARD};border:1px solid {card_border};
                        border-radius:14px;padding:16px 14px;{card_extra_style}'>
              <div style='font-family:DM Sans;font-size:0.78rem;color:{TEXT_PRIMARY};
                          font-weight:700;letter-spacing:1px;margin-bottom:10px;
                          text-align:center;'>{pair_label}</div>
              <div style='font-family:JetBrains Mono;font-size:1.9rem;font-weight:700;
                          color:{TEXT_PRIMARY};text-align:center;line-height:1.0;'>
                {pa.current_corr:+.2f}
              </div>
              <div style='font-family:JetBrains Mono;font-size:0.7rem;color:{TEXT_MUTED};
                          text-align:center;margin:4px 0 12px 0;'>
                z = <span style='color:{color}'>{pa.current_z:+.2f}</span>
                &nbsp;·&nbsp; mean {pa.mean:+.2f}
              </div>
              <div style='text-align:center;'>
                <span style='display:inline-block;padding:5px 14px;border-radius:14px;
                             background:{pill_bg};color:{pill_color};
                             font-family:DM Sans;font-size:0.72rem;font-weight:700;
                             text-transform:uppercase;letter-spacing:1.8px;'>
                  {severity}
                </span>
              </div>
            </div>
        """)


# ── Selected pair detail ───────────────────────────────────────────────────

st.markdown(section_header("Pair Detail"), unsafe_allow_html=True)

# Pre-select first non-Normal pair if any
sorted_analyses = sorted(analyses, key=lambda pa: -SEVERITY_ORDER[pa.severity])
default_pick = f"{sorted_analyses[0].pair[0]} / {sorted_analyses[0].pair[1]}"
labels = [f"{pa.pair[0]} / {pa.pair[1]}" for pa in analyses]
pick = st.selectbox("Pair", options=labels, index=labels.index(default_pick))
pa = analyses[labels.index(pick)]

# Main chart: 60-day rolling corr with bands
threshold_corr = pa.mean - 2.0 * pa.std
fig = go.Figure()

# Background safety/danger zones
fig.add_hrect(y0=threshold_corr, y1=1.0,
              fillcolor="rgba(0,230,118,0.05)", line_width=0, layer="below")
fig.add_hrect(y0=-1.0, y1=threshold_corr,
              fillcolor="rgba(255,23,68,0.07)", line_width=0, layer="below")

# Historical break shading
z_series = (pa.rolling_60 - pa.mean) / (pa.std + 1e-12)
break_mask = z_series <= z_threshold_for_history
if break_mask.any():
    runs = []
    in_run = False
    start = None
    idx = z_series.index
    for i, m in enumerate(break_mask.values):
        if m and not in_run:
            in_run = True
            start = idx[i]
        elif not m and in_run:
            in_run = False
            runs.append((start, idx[i - 1]))
    if in_run:
        runs.append((start, idx[-1]))
    for x0, x1 in runs:
        fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(255,23,68,0.10)",
                      line_width=0, layer="below")

# Mean line
fig.add_hline(y=pa.mean, line=dict(color=ACCENT_CYAN, width=1.2, dash="dash"),
              annotation_text=f"mean {pa.mean:+.2f}",
              annotation_position="right",
              annotation_font=dict(color=ACCENT_CYAN, family="JetBrains Mono", size=10))
fig.add_hline(y=threshold_corr, line=dict(color=ACCENT_RED, width=1.2, dash="dash"),
              annotation_text=f"-2σ {threshold_corr:+.2f}",
              annotation_position="right",
              annotation_font=dict(color=ACCENT_RED, family="JetBrains Mono", size=10))

# 60-day rolling correlation line
fig.add_trace(go.Scatter(
    x=pa.rolling_60.index, y=pa.rolling_60.values,
    mode="lines",
    line=dict(color=TEXT_PRIMARY, width=1.8),
    name="60d corr",
    hovertemplate="%{x|%Y-%m-%d}<br>%{y:+.3f}<extra></extra>",
))

# Current marker
fig.add_trace(go.Scatter(
    x=[pa.rolling_60.index[-1]], y=[pa.current_corr],
    mode="markers",
    marker=dict(color=pa.severity_color, size=12,
                line=dict(color=TEXT_PRIMARY, width=1.5)),
    name="current",
    hovertemplate=f"current {pa.current_corr:+.3f}<br>z = {pa.current_z:+.2f}<extra></extra>",
    showlegend=False,
))

layout = get_plotly_layout()
layout["height"] = 440
layout["margin"] = dict(l=50, r=80, t=20, b=40)
layout["showlegend"] = False
layout["yaxis"]["range"] = [-1.05, 1.05]
fig.update_layout(**layout)
st.plotly_chart(fig, use_container_width=True)

# 20d vs 60d comparison
left, right = st.columns(2)


def small_corr_fig(series: pd.Series, label: str, color: str) -> go.Figure:
    f = go.Figure()
    f.add_hrect(y0=0, y1=1, fillcolor="rgba(0,230,118,0.04)", line_width=0, layer="below")
    f.add_hrect(y0=-1, y1=0, fillcolor="rgba(255,23,68,0.04)", line_width=0, layer="below")
    f.add_trace(go.Scatter(
        x=series.index, y=series.values, mode="lines",
        line=dict(color=color, width=1.4),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:+.3f}<extra></extra>",
    ))
    sl = get_plotly_layout()
    sl["height"] = 220
    sl["margin"] = dict(l=40, r=20, t=40, b=40)
    sl["showlegend"] = False
    sl["yaxis"]["range"] = [-1.05, 1.05]
    sl["title"] = dict(
        text=f"<span style='font-family:DM Sans;color:{TEXT_MUTED};font-size:11px;"
             f"text-transform:uppercase;letter-spacing:2px;'>{label}</span>",
        x=0.02, xanchor="left", y=0.97, yanchor="top",
    )
    f.update_layout(**sl)
    return f


with left:
    st.plotly_chart(small_corr_fig(pa.rolling_20, "20-day rolling", ACCENT_AMBER), use_container_width=True)
with right:
    st.plotly_chart(small_corr_fig(pa.rolling_60, "60-day rolling", ACCENT_CYAN), use_container_width=True)

# Quick interpretation hint
recent_20 = pa.rolling_20.iloc[-1] if len(pa.rolling_20) else float("nan")
recent_60 = pa.rolling_60.iloc[-1]
gap = recent_20 - recent_60
if abs(gap) > 0.15:
    if gap < 0:
        hint = (f"20-day corr ({recent_20:+.2f}) sits well below the 60-day ({recent_60:+.2f}) - "
                f"this looks like a fresh decoupling, not a sustained shift yet.")
    else:
        hint = (f"20-day corr ({recent_20:+.2f}) is well above the 60-day ({recent_60:+.2f}) - "
                f"recent re-coupling may pull the 60-day higher in coming weeks.")
else:
    hint = (f"20-day ({recent_20:+.2f}) and 60-day ({recent_60:+.2f}) are aligned - "
            f"the regime, whatever it is, looks sustained.")
render_html(f"""
    <div style='font-family:DM Sans;font-size:0.85rem;color:{TEXT_SECONDARY};
                background:{BG_CARD};border:1px solid {BORDER};border-radius:10px;
                padding:12px 18px;margin-top:8px;'>{hint}</div>
""")


# ── Historical context (only when in break) ────────────────────────────────

if pa.severity != "Normal":
    st.markdown(section_header(f"Historical Context · {pa.pair[0]} / {pa.pair[1]}"), unsafe_allow_html=True)
    hb = historical_breaks(pa, severity_threshold=z_threshold_for_history)
    if hb.empty:
        render_html(f"<div style='color:{TEXT_MUTED};padding:18px;font-family:DM Sans;'>"
                    f"No previous instances found at z &lt;= {z_threshold_for_history:.1f} "
                    f"in this lookback window.</div>")
    else:
        a, b = pa.pair
        fwd_cols = [f"{a}+5d", f"{a}+10d", f"{a}+20d",
                    f"{b}+5d", f"{b}+10d", f"{b}+20d"]
        # Add average row
        avg_row = {"date": "AVERAGE", "z": float(hb["z"].mean())}
        for c in fwd_cols:
            avg_row[c] = float(hb[c].mean())
        display = pd.concat([hb, pd.DataFrame([avg_row])], ignore_index=True)

        def fmt_pct(v):
            if pd.isna(v): return ""
            return f"{v:+.2f}%"

        def color_signed(v):
            if isinstance(v, str) or pd.isna(v):
                return ""
            if v > 0:
                return f"color:{ACCENT_GREEN};font-weight:600;"
            if v < 0:
                return f"color:{ACCENT_RED};font-weight:600;"
            return f"color:{TEXT_MUTED};"

        def bold_avg(row):
            if row["date"] == "AVERAGE":
                return [f"background-color:{BG_CARD_HOVER};color:{TEXT_PRIMARY};"
                        f"border-top:2px solid {ACCENT_CYAN};font-weight:700;"] * len(row)
            return [""] * len(row)

        styled = (
            display.style
            .format({c: fmt_pct for c in fwd_cols})
            .format({"z": "{:+.2f}"})
            .map(color_signed, subset=fwd_cols)
            .apply(bold_avg, axis=1)
            .set_properties(**{
                "background-color": BG_PRIMARY,
                "color": TEXT_SECONDARY,
                "border": f"1px solid {BORDER}",
                "font-family": "JetBrains Mono, monospace",
                "font-size": "12px",
            })
            .set_table_styles([
                {"selector": "th", "props": [
                    ("background-color", BG_CARD),
                    ("color", TEXT_PRIMARY),
                    ("border", f"1px solid {BORDER}"),
                    ("font-family", "DM Sans, sans-serif"),
                    ("font-size", "11px"),
                    ("text-transform", "uppercase"),
                    ("letter-spacing", "1px"),
                    ("padding", "9px 6px"),
                ]},
                {"selector": "td", "props": [("padding", "7px 6px")]},
            ])
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        render_html(f"""
            <div style='font-family:DM Sans;font-size:0.78rem;color:{TEXT_MUTED};
                        font-style:italic;margin-top:10px;line-height:1.5;'>
              Past correlation breaks do not guarantee similar outcomes. Correlations
              can drift without signaling directional moves; treat the table as base-rate
              context, not a forecast.
            </div>
        """)


# ── Alert log ──────────────────────────────────────────────────────────────

with st.expander("Recent alert log", expanded=False):
    alerts = load_alerts()
    if not alerts:
        render_html(f"<div style='color:{TEXT_MUTED};padding:8px 0;'>No alerts logged yet.</div>")
    else:
        recent = list(reversed(alerts))[:30]
        rows_html = ""
        for a in recent:
            sev = a.get("severity", "?")
            _, sev_color = classify(-99 if sev == "Extreme" else (-2.25 if sev == "Significant" else (-1.75 if sev == "Notable" else 0)))
            ts = a.get("timestamp", "")
            try:
                ts_pretty = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M UTC")
            except Exception:
                ts_pretty = ts
            rows_html += f"""
                <div style='display:grid;grid-template-columns:170px 130px 90px 90px 1fr;gap:14px;
                            align-items:center;padding:8px 12px;border-bottom:1px solid {BORDER};
                            font-family:JetBrains Mono;font-size:0.78rem;color:{TEXT_SECONDARY};'>
                  <div>{ts_pretty}</div>
                  <div style='color:{TEXT_PRIMARY};font-weight:600;'>{a.get('pair','')}</div>
                  <div style='text-align:right;'>{a.get('current_corr', 0):+.2f}</div>
                  <div style='text-align:right;color:{sev_color};'>z {a.get('z_score', 0):+.2f}</div>
                  <div>
                    <span style='display:inline-block;padding:3px 10px;border-radius:10px;
                                 background:{sev_color}33;color:{sev_color};
                                 font-family:DM Sans;font-size:0.65rem;font-weight:700;
                                 text-transform:uppercase;letter-spacing:1.5px;'>{sev}</span>
                  </div>
                </div>
            """
        render_html(f"<div style='border-top:1px solid {BORDER};'>{rows_html}</div>")
        render_html(f"<div style='font-family:DM Sans;font-size:0.7rem;color:{TEXT_MUTED};"
                    f"padding-top:8px;'>Stored in <code>{ALERT_FILE}</code> · last 200 entries</div>")
