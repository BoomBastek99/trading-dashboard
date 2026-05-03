"""
Sentiment Analysis Dashboard.

Pulls news articles per ticker from NewsAPI (if a key is supplied) or Google
News RSS as fallback, scores each headline+snippet with VADER (boosted with a
finance lexicon), and shows a cockpit-style sentiment gauge per ticker plus
key-driver articles and an aggregate market-sentiment bar.

Run with:
    py -3.13 -m streamlit run sentiment_analysis_dashboard.py --server.port=8507
"""

from __future__ import annotations

import json
import textwrap
import urllib.parse
import urllib.request
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from zoneinfo import ZoneInfo

import nltk
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from design_system import *  # noqa: F401,F403

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Sentiment Briefing",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


def render_html(s: str) -> None:
    st.markdown(textwrap.dedent(s).strip(), unsafe_allow_html=True)


# ── VADER + finance lexicon boost ──────────────────────────────────────────

FINANCE_LEXICON = {
    # bullish
    "beats": 2.5, "beat": 2.0, "soars": 3.0, "soared": 3.0, "soar": 2.5,
    "surge": 2.5, "surges": 2.5, "surged": 2.5,
    "rally": 2.0, "rallies": 2.0, "rallied": 2.0, "rallying": 2.0,
    "gains": 1.5, "gained": 1.5, "gaining": 1.5,
    "rises": 1.2, "rose": 1.2, "rising": 1.2,
    "jumps": 2.0, "jumped": 2.0, "jumping": 2.0,
    "rebound": 2.0, "rebounds": 2.0,
    "bullish": 2.5, "outperform": 2.5, "outperforms": 2.5,
    "upgrade": 2.0, "upgrades": 2.0, "upgraded": 2.0,
    "boost": 1.5, "boosts": 1.5, "boosted": 1.5,
    "record": 1.5, "highs": 1.5, "all-time": 1.5,
    "strong": 1.5, "robust": 1.5, "solid": 1.0,
    "profit": 1.5, "profits": 1.5, "profitable": 2.0,
    "raises": 1.0, "raised": 1.0,
    # bearish
    "misses": -2.5, "miss": -2.0, "missed": -2.0,
    "plunge": -3.0, "plunges": -3.0, "plunged": -3.0, "plunging": -3.0,
    "tumble": -2.5, "tumbles": -2.5, "tumbled": -2.5,
    "crash": -3.0, "crashes": -3.0, "crashed": -3.0, "crashing": -3.0,
    "slump": -2.0, "slumps": -2.0, "slumped": -2.0,
    "falls": -1.5, "fell": -1.5, "falling": -1.5,
    "drops": -1.5, "dropped": -1.5, "dropping": -1.5,
    "sinks": -2.0, "sank": -2.0,
    "bearish": -2.5, "underperform": -2.5, "underperforms": -2.5,
    "downgrade": -2.0, "downgrades": -2.0, "downgraded": -2.0,
    "loss": -1.5, "losses": -1.5,
    "lows": -1.5, "weak": -1.5, "weakness": -1.5,
    "warning": -1.5, "warns": -1.5, "warned": -1.5,
    "concerns": -1.0, "concern": -1.0,
    "selloff": -2.5, "sell-off": -2.5,
    "lawsuit": -1.5, "probe": -1.0, "investigation": -1.0,
    "halts": -1.5, "halted": -1.5,
    "layoffs": -2.0, "layoff": -2.0, "cut": -1.0, "cuts": -1.0,
    "recession": -2.5, "bankruptcy": -3.5,
}

# Common ticker -> search alias for better news matching
TICKER_ALIASES = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SPY":     "S&P 500 ETF",
    "QQQ":     "Nasdaq 100 ETF",
    "GLD":     "gold ETF",
    "TLT":     "long bond ETF",
}


@st.cache_resource(show_spinner=False)
def get_analyzer() -> SentimentIntensityAnalyzer:
    try:
        sia = SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(FINANCE_LEXICON)
    return sia


# ── HTML stripping helper ──────────────────────────────────────────────────

class _Stripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._chunks: list[str] = []
    def handle_data(self, data):
        self._chunks.append(data)
    @property
    def text(self) -> str:
        return " ".join("".join(self._chunks).split())


def strip_html(s: str) -> str:
    p = _Stripper()
    try:
        p.feed(s)
    except Exception:
        return s
    return p.text


# ── News fetchers ──────────────────────────────────────────────────────────

USER_AGENT = "Mozilla/5.0 (compatible; SentimentDashboard/1.0)"


def _http_get(url: str, timeout: int = 12) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_google_news_rss(query: str, lookback_days: int) -> list[dict]:
    q = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    try:
        content = _http_get(url)
    except Exception as exc:
        raise RuntimeError(f"Google News fetch failed: {exc}") from exc
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise RuntimeError(f"Google News RSS parse failed: {exc}") from exc

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    out: list[dict] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_raw = (item.findtext("pubDate") or "").strip()
        desc_raw = (item.findtext("description") or "").strip()
        src_elem = item.find("source")
        source = src_elem.text.strip() if (src_elem is not None and src_elem.text) else "Google News"

        try:
            pub_date = parsedate_to_datetime(pub_raw)
            if pub_date.tzinfo is None:
                pub_date = pub_date.replace(tzinfo=timezone.utc)
        except Exception:
            pub_date = datetime.now(timezone.utc)

        if pub_date < cutoff:
            continue

        snippet = strip_html(desc_raw)[:300]
        # Google News titles often include " - Source" suffix; strip that for cleanliness
        clean_title = title
        if " - " in title:
            head, sep, tail = title.rpartition(" - ")
            if 0 < len(tail) < 50:
                clean_title = head
                if source == "Google News":
                    source = tail

        out.append({
            "title": clean_title,
            "source": source,
            "published": pub_date,
            "snippet": snippet,
            "link": link,
        })
    return out


def fetch_newsapi(query: str, lookback_days: int, api_key: str) -> list[dict]:
    from_date = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    params = urllib.parse.urlencode({
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 50,
        "apiKey": api_key,
    })
    url = f"https://newsapi.org/v2/everything?{params}"
    try:
        raw = _http_get(url)
    except Exception as exc:
        raise RuntimeError(f"NewsAPI fetch failed: {exc}") from exc
    try:
        data = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"NewsAPI parse failed: {exc}") from exc
    if data.get("status") != "ok":
        raise RuntimeError(f"NewsAPI: {data.get('message', 'unknown error')}")

    out = []
    for art in data.get("articles", []):
        pub_raw = art.get("publishedAt") or ""
        try:
            pub_date = datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
        except Exception:
            pub_date = datetime.now(timezone.utc)
        out.append({
            "title": (art.get("title") or "").strip(),
            "source": ((art.get("source") or {}).get("name") or "NewsAPI").strip(),
            "published": pub_date,
            "snippet": (art.get("description") or "").strip()[:300],
            "link": art.get("url") or "",
        })
    return out


@st.cache_data(ttl=900, show_spinner=False)  # 15-min cache
def fetch_articles(ticker: str, lookback_days: int, newsapi_key: str | None) -> list[dict]:
    query = TICKER_ALIASES.get(ticker, f"{ticker} stock")
    if newsapi_key:
        try:
            return fetch_newsapi(query, lookback_days, newsapi_key)
        except Exception:
            pass  # silently fall back
    return fetch_google_news_rss(query, lookback_days)


# ── Scoring + aggregation ──────────────────────────────────────────────────

def score_articles(articles: list[dict], sia: SentimentIntensityAnalyzer) -> list[dict]:
    for a in articles:
        text = (a.get("title") or "") + ". " + (a.get("snippet") or "")
        a["sentiment"] = float(sia.polarity_scores(text)["compound"])
    return articles


def aggregate(articles: list[dict], recent_hours: float = 24.0) -> dict:
    """Recency-weighted average sentiment + momentum vs >3-day-old baseline."""
    if not articles:
        return {"current": 0.0, "momentum": 0.0, "n": 0, "n_recent": 0}

    now = datetime.now(timezone.utc)
    recent_thresh = timedelta(hours=recent_hours)
    momentum_thresh = timedelta(days=3)

    weighted_sum = 0.0
    weight_total = 0.0
    n_recent = 0
    older_scores: list[float] = []

    for a in articles:
        age = now - a["published"]
        weight = 2.0 if age < recent_thresh else 1.0
        weighted_sum += a["sentiment"] * weight
        weight_total += weight
        if age < recent_thresh:
            n_recent += 1
        if age > momentum_thresh:
            older_scores.append(a["sentiment"])

    current = weighted_sum / weight_total if weight_total > 0 else 0.0
    if older_scores:
        baseline = sum(older_scores) / len(older_scores)
        momentum = current - baseline
    else:
        momentum = 0.0

    return {"current": current, "momentum": momentum, "n": len(articles), "n_recent": n_recent}


def sentiment_color(score: float) -> str:
    if score > 0.15:
        return ACCENT_GREEN
    if score < -0.15:
        return ACCENT_RED
    return TEXT_SECONDARY


# ── Sidebar ────────────────────────────────────────────────────────────────

DEFAULT_TICKERS = ["SPY", "AAPL", "NVDA", "TSLA", "BTC-USD"]

with st.sidebar:
    st.markdown(section_header("Briefing Inputs"), unsafe_allow_html=True)
    tickers_str = st.text_input("Tickers (comma separated)", value=",".join(DEFAULT_TICKERS))
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))[:8]  # de-dupe + cap at 8 for layout

    lookback = st.slider("Lookback (days)", 1, 14, 3)

    st.markdown(section_header("NewsAPI (optional)"), unsafe_allow_html=True)
    st.caption("Get a free key at newsapi.org. Falls back to Google News RSS if blank.")
    newsapi_key = st.text_input("API key", value="", type="password")

    refresh = st.button("Refresh", type="primary", use_container_width=True)


# ── Run ─────────────────────────────────────────────────────────────────────

if not tickers:
    render_html(f"<div style='color:{TEXT_MUTED};padding:64px 0;text-align:center;'>"
                f"Add at least one ticker in the sidebar.</div>")
    st.stop()

if refresh:
    fetch_articles.clear()

sia = get_analyzer()

per_ticker: dict[str, dict] = {}
fetch_errors: dict[str, str] = {}

with st.spinner("Loading news..."):
    progress = st.progress(0.0, text="Starting...")
    for i, t in enumerate(tickers):
        progress.progress(i / len(tickers), text=f"Fetching {t}...")
        try:
            arts = fetch_articles(t, lookback, newsapi_key.strip() or None)
            arts = score_articles(arts, sia)
            arts.sort(key=lambda a: a["published"], reverse=True)
            agg = aggregate(arts)
            per_ticker[t] = {"articles": arts, **agg}
        except Exception as exc:
            fetch_errors[t] = str(exc)
            per_ticker[t] = {"articles": [], "current": 0.0, "momentum": 0.0, "n": 0, "n_recent": 0}
    progress.empty()


# ── Top bar ────────────────────────────────────────────────────────────────

now_et = pd.Timestamp.now(tz=ZoneInfo("America/New_York"))
date_str = now_et.strftime("%A, %B %d, %Y")
time_str = now_et.strftime("%H:%M ET")
total_articles = sum(d["n"] for d in per_ticker.values())

render_html(f"""
    <div style='border-bottom:1px solid {BORDER};padding-bottom:14px;margin-bottom:18px;
                display:flex;align-items:flex-end;justify-content:space-between;gap:20px;flex-wrap:wrap;'>
      <div>
        <div style='font-family:DM Sans;font-size:0.7rem;color:{ACCENT_CYAN};
                    text-transform:uppercase;letter-spacing:5px;margin-bottom:4px;'>
          {status_dot('connected')}Sentiment Briefing
        </div>
        <div style='font-family:DM Sans;font-size:1.7rem;color:{TEXT_PRIMARY};font-weight:700;
                    letter-spacing:1px;line-height:1.1;'>
          MARKET SENTIMENT BRIEFING
        </div>
        <div style='font-family:JetBrains Mono;font-size:0.85rem;color:{TEXT_MUTED};margin-top:4px;'>
          {date_str} &nbsp;·&nbsp; {time_str}
        </div>
      </div>
      <div style='text-align:right;font-family:JetBrains Mono;font-size:0.85rem;color:{TEXT_SECONDARY};'>
        <div style='color:{TEXT_PRIMARY};font-size:1.3rem;font-weight:700;'>
          {total_articles}
        </div>
        <div style='color:{TEXT_MUTED};font-size:0.7rem;text-transform:uppercase;letter-spacing:2px;'>
          articles · {len(tickers)} tickers · {lookback}d window
        </div>
      </div>
    </div>
""")

if fetch_errors:
    bullets = "".join(f"<li>{t}: {e}</li>" for t, e in fetch_errors.items())
    render_html(f"""
        <div style='background:rgba(255,193,7,0.08);border:1px solid {ACCENT_AMBER};
                    border-left:3px solid {ACCENT_AMBER};border-radius:10px;
                    padding:12px 18px;margin-bottom:14px;
                    font-family:DM Sans;font-size:0.82rem;color:{TEXT_SECONDARY};'>
          <b style='color:{ACCENT_AMBER};text-transform:uppercase;letter-spacing:2px;font-size:0.7rem;'>
            news fetch errors</b>
          <ul style='margin:6px 0 0 0;padding-left:18px;'>{bullets}</ul>
        </div>
    """)


# ── Sentiment gauge per ticker ─────────────────────────────────────────────

def sentiment_gauge_fig(score: float) -> go.Figure:
    color = sentiment_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={
            "valueformat": "+.2f",
            "font": {"family": "JetBrains Mono", "size": 36, "color": color},
        },
        gauge={
            "shape": "angular",
            "axis": {
                "range": [-1, 1],
                "tickvals": [-1, -0.5, 0, 0.5, 1],
                "ticktext": ["-1", "-.5", "0", "+.5", "+1"],
                "tickwidth": 1,
                "tickcolor": TEXT_MUTED,
                "tickfont": {"family": "JetBrains Mono", "color": TEXT_MUTED, "size": 9},
            },
            "bar": {"color": "rgba(0,0,0,0)", "thickness": 0},
            "bgcolor": BG_CARD,
            "borderwidth": 0,
            "steps": [
                {"range": [-1.0, -0.3], "color": "rgba(255,23,68,0.30)"},
                {"range": [-0.3,  0.3], "color": "rgba(138,138,154,0.18)"},
                {"range": [ 0.3,  1.0], "color": "rgba(0,230,118,0.30)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 5},
                "thickness": 0.95,
                "value": score,
            },
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    layout = get_plotly_layout()
    layout["height"] = 220
    layout["margin"] = dict(l=10, r=10, t=10, b=10)
    fig.update_layout(**layout)
    return fig


def momentum_label(momentum: float) -> tuple[str, str, str]:
    if momentum > 0.05:
        return "▲", "Improving", ACCENT_GREEN
    if momentum < -0.05:
        return "▼", "Declining", ACCENT_RED
    return "—", "Stable", TEXT_SECONDARY


cols = st.columns(len(tickers))
for i, t in enumerate(tickers):
    d = per_ticker[t]
    color = sentiment_color(d["current"])
    arrow, mom_text, mom_color = momentum_label(d["momentum"])
    with cols[i]:
        render_html(f"""
            <div style='background:{BG_CARD};border:1px solid {BORDER};border-radius:14px;
                        padding:18px 14px 12px 14px;
                        box-shadow:0 0 24px rgba(0,212,255,0.04);'>
              <div style='display:flex;align-items:baseline;justify-content:space-between;
                          margin-bottom:8px;'>
                <span style='font-family:DM Sans;font-size:1.4rem;font-weight:700;
                             color:{TEXT_PRIMARY};letter-spacing:1px;'>{t}</span>
                <span style='font-family:JetBrains Mono;font-size:0.7rem;color:{TEXT_MUTED};'>
                  {d['n']} art</span>
              </div>
            </div>
        """)
        st.plotly_chart(sentiment_gauge_fig(d["current"]), use_container_width=True)
        render_html(f"""
            <div style='display:flex;align-items:center;justify-content:center;gap:8px;
                        margin-top:-12px;margin-bottom:8px;
                        font-family:DM Sans;font-size:0.85rem;'>
              <span style='color:{mom_color};font-size:1.2rem;font-weight:700;'>{arrow}</span>
              <span style='color:{mom_color};font-weight:600;'>{mom_text}</span>
              <span style='color:{TEXT_MUTED};font-family:JetBrains Mono;font-size:0.72rem;'>
                ({d['momentum']:+.2f})</span>
            </div>
            <div style='text-align:center;color:{TEXT_MUTED};font-family:JetBrains Mono;
                        font-size:0.7rem;'>
              {d['n_recent']} recent · {d['n']} total
            </div>
        """)


# ── Detail tabs (key drivers per ticker) ───────────────────────────────────

st.markdown(section_header("Key Drivers"), unsafe_allow_html=True)

tabs = st.tabs(tickers)
for i, t in enumerate(tickers):
    d = per_ticker[t]
    with tabs[i]:
        if not d["articles"]:
            err = fetch_errors.get(t, "no articles in lookback window")
            render_html(f"<div style='color:{TEXT_MUTED};padding:22px;font-family:DM Sans;'>"
                        f"No articles for <b>{t}</b> ({err}).</div>")
            continue

        top5 = sorted(d["articles"], key=lambda a: abs(a["sentiment"]), reverse=True)[:5]
        for a in top5:
            s = a["sentiment"]
            border_col = ACCENT_GREEN if s > 0.15 else (ACCENT_RED if s < -0.15 else TEXT_MUTED)
            dot_col = sentiment_color(s)
            pub_str = a["published"].astimezone(ZoneInfo("America/New_York")).strftime("%b %d · %H:%M ET")
            link = a.get("link") or "#"
            title_html = (
                f"<a href='{link}' target='_blank' rel='noopener' "
                f"style='color:{TEXT_PRIMARY};text-decoration:none;'>{a['title']}</a>"
            )
            snippet = a.get("snippet") or ""
            render_html(f"""
                <div style='background:{BG_CARD};border:1px solid {BORDER};
                            border-left:3px solid {border_col};border-radius:10px;
                            padding:14px 18px;margin-bottom:10px;'>
                  <div style='display:flex;align-items:center;justify-content:space-between;gap:14px;
                              font-family:JetBrains Mono;font-size:0.7rem;color:{TEXT_MUTED};
                              margin-bottom:6px;'>
                    <span><b style='color:{TEXT_SECONDARY}'>{a['source']}</b> &nbsp;·&nbsp; {pub_str}</span>
                    <span style='display:inline-flex;align-items:center;gap:6px;color:{dot_col};'>
                      <span style='display:inline-block;width:8px;height:8px;border-radius:50%;
                                   background:{dot_col};box-shadow:0 0 8px {dot_col}80;'></span>
                      {s:+.2f}
                    </span>
                  </div>
                  <div style='font-family:DM Sans;font-size:0.95rem;font-weight:600;color:{TEXT_PRIMARY};
                              line-height:1.4;margin-bottom:6px;'>{title_html}</div>
                  <div style='font-family:DM Sans;font-size:0.8rem;color:{TEXT_MUTED};line-height:1.5;'>
                    {snippet}
                  </div>
                </div>
            """)


# ── Aggregate horizontal bar ───────────────────────────────────────────────

st.markdown(section_header("Aggregate Market Sentiment"), unsafe_allow_html=True)

rows = [(t, per_ticker[t]["current"]) for t in tickers if per_ticker[t]["n"] > 0]
if not rows:
    render_html(f"<div style='color:{TEXT_MUTED};padding:18px;'>No sentiment data yet.</div>")
else:
    rows.sort(key=lambda kv: kv[1])  # ascending: most bearish first
    bear = [(t, s) for t, s in rows if s < 0]
    bull = [(t, s) for t, s in rows if s >= 0]

    fig = go.Figure()
    # Bearish: stack from 0 going LEFT (most bearish furthest left)
    cum_left = 0.0
    for t, s in sorted(bear, key=lambda kv: kv[1]):  # most negative first → furthest left
        width = abs(s)
        cum_left -= width
        fig.add_trace(go.Bar(
            y=["Market"], x=[width], base=[cum_left],
            orientation="h",
            marker=dict(color=ACCENT_RED, line=dict(color=BG_PRIMARY, width=2)),
            text=[f"<b>{t}</b><br>{s:+.2f}"], textposition="inside",
            insidetextfont=dict(color=TEXT_PRIMARY, family="JetBrains Mono", size=11),
            hovertemplate=f"{t}<br>%{{x:.2f}}<extra></extra>",
            showlegend=False, name=t,
        ))
    # Bullish: stack from 0 going RIGHT (most bullish furthest right)
    cum_right = 0.0
    for t, s in sorted(bull, key=lambda kv: kv[1]):  # least positive first → closest to 0
        width = s
        fig.add_trace(go.Bar(
            y=["Market"], x=[width], base=[cum_right],
            orientation="h",
            marker=dict(color=ACCENT_GREEN, line=dict(color=BG_PRIMARY, width=2)),
            text=[f"<b>{t}</b><br>{s:+.2f}"], textposition="inside",
            insidetextfont=dict(color=BG_PRIMARY, family="JetBrains Mono", size=11),
            hovertemplate=f"{t}<br>%{{x:.2f}}<extra></extra>",
            showlegend=False, name=t,
        ))
        cum_right += width

    extent = max(abs(cum_left), abs(cum_right), 0.5) * 1.15

    layout = get_plotly_layout()
    layout["height"] = 140
    layout["margin"] = dict(l=20, r=20, t=10, b=30)
    layout["barmode"] = "relative"
    layout["bargap"] = 0.0
    layout["xaxis"]["range"] = [-extent, extent]
    layout["xaxis"]["zeroline"] = True
    layout["xaxis"]["zerolinecolor"] = TEXT_PRIMARY
    layout["xaxis"]["zerolinewidth"] = 2
    layout["yaxis"]["showticklabels"] = False
    layout["yaxis"]["showgrid"] = False
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# ── Disclaimer ─────────────────────────────────────────────────────────────

render_html(f"""
    <div style='margin-top:18px;padding:12px 18px;border-top:1px solid {BORDER};
                font-family:DM Sans;font-size:0.75rem;color:{TEXT_MUTED};line-height:1.6;
                font-style:italic;'>
      Sentiment scores are based on automated text analysis (VADER + finance lexicon)
      and may misinterpret context, sarcasm, or domain-specific phrasing. Use as one
      input among many.
    </div>
""")
