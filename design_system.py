import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Color constants
BG_PRIMARY = "#0a0a0f"
BG_CARD = "#12121a"
BG_CARD_HOVER = "#1a1a24"
BORDER = "rgba(255,255,255,0.06)"
ACCENT_CYAN = "#00d4ff"
ACCENT_GREEN = "#00e676"
ACCENT_RED = "#ff1744"
ACCENT_AMBER = "#ffc107"
ACCENT_VIOLET = "#7c4dff"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#8a8a9a"
TEXT_MUTED = "#5a5a6a"
GRID_LINE = "rgba(255,255,255,0.04)"

REGIME_COLORS = {
    "Low Vol": ACCENT_GREEN,
    "Bull": ACCENT_GREEN,
    "Medium Vol": ACCENT_AMBER,
    "Neutral": ACCENT_AMBER,
    "High Vol": ACCENT_RED,
    "Bear": ACCENT_RED,
    "Uncertain": ACCENT_VIOLET
}

def apply_theme():
    """Inject custom CSS for dark trading terminal aesthetic."""
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=DM+Sans:wght@400;700&display=swap');

    * {{
        font-family: 'DM Sans', sans-serif;
    }}

    .metric {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }}

    body {{
        background-color: {BG_PRIMARY} !important;
        color: {TEXT_PRIMARY} !important;
        margin: 0 !important;
        padding: 0 !important;
    }}

    .main {{
        background-color: {BG_PRIMARY} !important;
        padding: 0 !important;
    }}

    .sidebar .sidebar-content {{
        background-color: {BG_CARD} !important;
    }}

    .stApp {{
        background-color: {BG_PRIMARY} !important;
    }}

    /* Hide Streamlit elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .css-1rs6os {{visibility: hidden;}} /* Made with Streamlit */
    .css-14xtw13 {{visibility: hidden;}} /* Footer */

    /* Top accent line */
    .main::before {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, {ACCENT_CYAN}, transparent);
        z-index: 9999;
    }}

    /* Card styling */
    .card {{
        background-color: {BG_CARD} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 12px !important;
        padding: 20px !important;
        margin: 10px 0 !important;
    }}

    .card:hover {{
        background-color: {BG_CARD_HOVER} !important;
    }}

    /* Remove default padding */
    .block-container {{
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }}

    /* Dataframe styling */
    .dataframe {{
        font-family: 'JetBrains Mono', monospace !important;
        background-color: {BG_PRIMARY} !important;
        color: {TEXT_SECONDARY} !important;
    }}

    .dataframe th {{
        background-color: {BG_CARD} !important;
        color: {TEXT_PRIMARY} !important;
        border: 1px solid {BORDER} !important;
    }}

    .dataframe td {{
        background-color: {BG_PRIMARY} !important;
        color: {TEXT_SECONDARY} !important;
        border: 1px solid {BORDER} !important;
    }}

    /* Plotly styling */
    .js-plotly-plot {{
        background-color: {BG_PRIMARY} !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def metric_card(label, value, color=ACCENT_CYAN):
    """Return styled HTML for a metric display."""
    html = f"""
    <div style="
        background-color: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    ">
        <div style="
            font-family: 'JetBrains Mono', monospace;
            font-size: 2.5rem;
            font-weight: 700;
            color: {color};
            margin-bottom: 5px;
        ">{value}</div>
        <div style="
            font-size: 0.8rem;
            color: {TEXT_MUTED};
            text-transform: uppercase;
            letter-spacing: 1px;
        ">{label}</div>
    </div>
    """
    return html

def regime_badge(regime_name, confidence=None):
    """Return styled HTML pill/badge for regime."""
    color = REGIME_COLORS.get(regime_name, ACCENT_VIOLET)
    bg_color = color + "33"  # 20% opacity
    confidence_text = f" ({confidence}%)" if confidence else ""
    html = f"""
    <span style="
        background-color: {bg_color};
        color: {color};
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        box-shadow: 0 0 10px {color}40;
        display: inline-block;
        margin: 5px;
    ">{regime_name}{confidence_text}</span>
    """
    return html

def section_header(text):
    """Return styled HTML for section headers."""
    html = f"""
    <div style="
        display: flex;
        align-items: center;
        margin: 20px 0 10px 0;
    ">
        <span style="
            font-size: 0.7rem;
            color: {TEXT_MUTED};
            text-transform: uppercase;
            letter-spacing: 3px;
            font-weight: 600;
        ">{text}</span>
        <hr style="
            flex: 1;
            border: none;
            height: 1px;
            background-color: {BORDER};
            margin-left: 10px;
        ">
    </div>
    """
    return html

def status_dot(status):
    """Return a small colored dot for status."""
    if status in ["connected", "active"]:
        color = ACCENT_GREEN
    elif status in ["disconnected", "error"]:
        color = ACCENT_RED
    elif status == "warning":
        color = ACCENT_AMBER
    else:
        color = TEXT_MUTED
    
    html = f"""
    <span style="
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: {color};
        margin-right: 5px;
    "></span>
    """
    return html

def pnl_color(value):
    """Return color based on P&L value."""
    if value > 0:
        return ACCENT_GREEN
    elif value < 0:
        return ACCENT_RED
    else:
        return TEXT_SECONDARY

def get_plotly_layout():
    """Return base Plotly layout dict."""
    return {
        "paper_bgcolor": BG_PRIMARY,
        "plot_bgcolor": BG_PRIMARY,
        "font": {
            "family": "JetBrains Mono",
            "color": TEXT_SECONDARY
        },
        "xaxis": {
            "gridcolor": GRID_LINE,
            "linecolor": BG_PRIMARY,
            "tickcolor": TEXT_MUTED,
            "tickfont": {"color": TEXT_MUTED}
        },
        "yaxis": {
            "gridcolor": GRID_LINE,
            "linecolor": BG_PRIMARY,
            "tickcolor": TEXT_MUTED,
            "tickfont": {"color": TEXT_MUTED}
        },
        "hoverlabel": {
            "bgcolor": BG_CARD,
            "bordercolor": ACCENT_CYAN,
            "font": {"color": TEXT_PRIMARY}
        },
        "margin": {"l": 40, "r": 20, "t": 40, "b": 40}
    }

def style_dataframe(df):
    """Apply dark styling to pandas dataframe."""
    styles = [
        {
            "selector": "table",
            "props": [
                ("background-color", BG_PRIMARY),
                ("color", TEXT_SECONDARY),
                ("font-family", "JetBrains Mono, monospace"),
                ("border-collapse", "collapse")
            ]
        },
        {
            "selector": "th",
            "props": [
                ("background-color", BG_CARD),
                ("color", TEXT_PRIMARY),
                ("border", f"1px solid {BORDER}"),
                ("padding", "8px")
            ]
        },
        {
            "selector": "td",
            "props": [
                ("background-color", BG_PRIMARY),
                ("color", TEXT_SECONDARY),
                ("border", f"1px solid {BORDER}"),
                ("padding", "8px")
            ]
        },
        {
            "selector": "tr:hover",
            "props": [("background-color", BG_CARD_HOVER)]
        }
    ]
    return df.style.set_table_styles(styles)