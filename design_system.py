"""
Shared design system for the Trading Dashboard Suite.

Usage:
    from design_system import *
    apply_theme()
"""

import streamlit as st
import pandas as pd

# ── Color Constants ──────────────────────────────────────────────────────────

BG_PRIMARY       = "#0a0a0f"
BG_CARD          = "#12121a"
BG_CARD_HOVER    = "#1a1a24"
BORDER           = "rgba(255,255,255,0.06)"

ACCENT_CYAN      = "#00d4ff"
ACCENT_GREEN     = "#00e676"
ACCENT_RED       = "#ff1744"
ACCENT_AMBER     = "#ffc107"
ACCENT_VIOLET    = "#7c4dff"

TEXT_PRIMARY     = "#ffffff"
TEXT_SECONDARY   = "#8a8a9a"
TEXT_MUTED       = "#5a5a6a"

GRID_LINE        = "rgba(255,255,255,0.04)"

REGIME_COLORS = {
    "Low Vol":    ACCENT_GREEN,
    "Bull":       ACCENT_GREEN,
    "Medium Vol": ACCENT_AMBER,
    "Neutral":    ACCENT_AMBER,
    "High Vol":   ACCENT_RED,
    "Bear":       ACCENT_RED,
    "Uncertain":  ACCENT_VIOLET,
}


# ── Theme Injection ─────────────────────────────────────────────────────────

def apply_theme():
    """Inject custom CSS into the Streamlit app for a dark trading-terminal look."""
    st.markdown(
        f"""
        <style>
        /* ── Fonts ─────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'DM Sans', sans-serif;
        }}

        /* ── App Background ────────────────────────────── */
        .stApp {{
            background-color: {BG_PRIMARY};
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] > div:first-child {{
            background-color: {BG_CARD};
        }}

        /* ── Hide Streamlit Chrome ─────────────────────── */
        /* Hide the hamburger menu, footer and Deploy button, but keep the
           header element interactive so the sidebar-collapse toggle works. */
        #MainMenu                              {{ visibility: hidden; }}
        footer                                 {{ visibility: hidden; }}
        .stAppDeployButton                     {{ display: none !important; }}
        [data-testid="stToolbar"]              {{ display: none !important; }}
        [data-testid="stStatusWidget"]         {{ display: none !important; }}
        header[data-testid="stHeader"] {{
            background: transparent !important;
            height: 0 !important;
        }}
        /* Force the sidebar-expand control to stay visible/clickable when
           the sidebar is collapsed. */
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"] {{
            visibility: visible !important;
            display: block !important;
            z-index: 100000 !important;
        }}

        /* ── Cyan accent line at top ───────────────────── */
        .stApp::before {{
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, {ACCENT_CYAN}, transparent 70%);
            z-index: 9999;
            pointer-events: none;
        }}

        /* ── Edge-to-edge: reduce default padding ──────── */
        .block-container {{
            padding: 2rem 1.5rem 2rem 1.5rem !important;
            max-width: 100% !important;
        }}

        /* ── Metric elements ───────────────────────────── */
        [data-testid="stMetricValue"] {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 2rem;
            font-weight: 700;
        }}
        [data-testid="stMetricLabel"] {{
            font-family: 'DM Sans', sans-serif;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: {TEXT_MUTED};
        }}

        /* ── Generic card / container styling ──────────── */
        div[data-testid="stExpander"],
        div[data-testid="stVerticalBlock"] > div[data-testid="element-container"] > div > div > div {{
            border-radius: 12px;
        }}

        /* ── Dataframe chrome ──────────────────────────── */
        .stDataFrame {{
            border-radius: 12px;
            overflow: hidden;
        }}

        /* ── Tab bar ───────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: {BG_CARD};
            border-radius: 8px;
            color: {TEXT_SECONDARY};
            border: 1px solid {BORDER};
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {BG_CARD_HOVER};
            color: {ACCENT_CYAN};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Helper Functions ─────────────────────────────────────────────────────────

def metric_card(label: str, value: str, color: str = ACCENT_CYAN) -> str:
    """Return styled HTML for a metric card with a large colored number and muted label."""
    return f"""
    <div style="
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 22px 18px;
        text-align: center;
    ">
        <div style="
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.2rem;
            font-weight: 700;
            color: {color};
            line-height: 1.1;
            margin-bottom: 6px;
        ">{value}</div>
        <div style="
            font-family: 'DM Sans', sans-serif;
            font-size: 0.7rem;
            color: {TEXT_MUTED};
            text-transform: uppercase;
            letter-spacing: 1.5px;
        ">{label}</div>
    </div>
    """


def regime_badge(regime_name: str, confidence: int | float | None = None) -> str:
    """Return styled HTML pill/badge for a regime label.

    ``confidence`` should be 0-100 (percentage) or ``None`` to omit.
    """
    color = REGIME_COLORS.get(regime_name, ACCENT_VIOLET)
    # 20 % opacity background  →  hex alpha 33
    bg = color + "33"
    conf_html = f"&nbsp;({confidence:.0f}%)" if confidence is not None else ""
    return f"""
    <span style="
        display: inline-block;
        background: {bg};
        color: {color};
        padding: 5px 14px;
        border-radius: 20px;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
        box-shadow: 0 0 12px {color}40;
        white-space: nowrap;
    ">{regime_name}{conf_html}</span>
    """


def section_header(text: str) -> str:
    """Return styled HTML for a section header with a line extending to the right."""
    return f"""
    <div style="
        display: flex;
        align-items: center;
        margin: 28px 0 12px 0;
        gap: 12px;
    ">
        <span style="
            font-family: 'DM Sans', sans-serif;
            font-size: 11px;
            font-weight: 600;
            color: {TEXT_MUTED};
            text-transform: uppercase;
            letter-spacing: 3px;
            white-space: nowrap;
        ">{text}</span>
        <span style="
            flex: 1;
            height: 1px;
            background: {BORDER};
        "></span>
    </div>
    """


def status_dot(status: str) -> str:
    """Return a small colored dot HTML span.

    Recognised statuses: connected, active → green ·
    disconnected, error → red · warning → amber.
    """
    _map = {
        "connected": ACCENT_GREEN, "active": ACCENT_GREEN,
        "disconnected": ACCENT_RED, "error": ACCENT_RED,
        "warning": ACCENT_AMBER,
    }
    color = _map.get(status.lower(), TEXT_MUTED)
    return (
        f'<span style="display:inline-block;width:8px;height:8px;'
        f'border-radius:50%;background:{color};margin-right:6px;'
        f'box-shadow:0 0 6px {color}80;vertical-align:middle;"></span>'
    )


def pnl_color(value: float) -> str:
    """Return green for positive, red for negative, muted for zero."""
    if value > 0:
        return ACCENT_GREEN
    if value < 0:
        return ACCENT_RED
    return TEXT_SECONDARY


# ── Plotly Layout ────────────────────────────────────────────────────────────

def get_plotly_layout() -> dict:
    """Return a base Plotly layout dict matching the dark trading-terminal theme."""
    return dict(
        paper_bgcolor=BG_PRIMARY,
        plot_bgcolor=BG_PRIMARY,
        font=dict(family="JetBrains Mono", color=TEXT_SECONDARY, size=12),
        xaxis=dict(
            gridcolor=GRID_LINE,
            linecolor=BG_PRIMARY,
            zerolinecolor=GRID_LINE,
            tickfont=dict(color=TEXT_MUTED),
        ),
        yaxis=dict(
            gridcolor=GRID_LINE,
            linecolor=BG_PRIMARY,
            zerolinecolor=GRID_LINE,
            tickfont=dict(color=TEXT_MUTED),
        ),
        hoverlabel=dict(
            bgcolor=BG_CARD,
            bordercolor=ACCENT_CYAN,
            font=dict(color=TEXT_PRIMARY, family="JetBrains Mono"),
        ),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_SECONDARY),
        ),
    )


# ── DataFrame Styling ────────────────────────────────────────────────────────

def style_dataframe(df: pd.DataFrame):
    """Apply dark styling to a pandas DataFrame for display in Streamlit.

    Returns a ``Styler`` object — pass it directly to ``st.dataframe()``.
    """
    return (
        df.style
        .set_properties(**{
            "background-color": BG_PRIMARY,
            "color": TEXT_SECONDARY,
            "border": f"1px solid {BORDER}",
            "font-family": "JetBrains Mono, monospace",
            "font-size": "13px",
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", BG_CARD),
                ("color", TEXT_PRIMARY),
                ("border", f"1px solid {BORDER}"),
                ("font-family", "DM Sans, sans-serif"),
                ("font-size", "12px"),
                ("text-transform", "uppercase"),
                ("letter-spacing", "1px"),
                ("padding", "10px 8px"),
            ]},
            {"selector": "td", "props": [
                ("padding", "8px"),
            ]},
            {"selector": "tr:hover td", "props": [
                ("background-color", BG_CARD_HOVER),
            ]},
        ])
    )
