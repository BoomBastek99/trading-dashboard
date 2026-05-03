"""
Unified market-data provider with fallback chain.

Order: FMP (primary, paid) → TwelveData (fallback, low quota) → yfinance (last resort).
All public functions return OHLCV DataFrames with the standard yfinance-style
columns ['Open','High','Low','Close','Volume'] and a DatetimeIndex sorted
ascending. Caching is centralized here so multipage / multi-dashboard
deployments share a single tile of fetched data.

Secrets are read from st.secrets (Streamlit Cloud) with env var fallback so
local CLI tests work too. See .streamlit/secrets.toml.example for the keys.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
import warnings
from typing import Callable

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

USER_AGENT = "TradingDashboards/1.0"
HTTP_TIMEOUT = 15

# Crypto base symbols handled by the FMP /historical-price-full endpoint as XXXUSD
# (yfinance uses XXX-USD; we translate transparently).
_CRYPTO_BASES = {
    "BTC", "ETH", "SOL", "DOGE", "XRP", "ADA", "LTC", "BCH", "MATIC", "DOT",
    "AVAX", "LINK", "UNI", "ATOM", "XLM", "TRX",
}


# ── Secret access ──────────────────────────────────────────────────────────

def get_secret(key: str, default: str = "") -> str:
    """st.secrets first, then environment variable, then default."""
    try:
        val = st.secrets.get(key)
        if val:
            return str(val)
    except (FileNotFoundError, KeyError, AttributeError, Exception):
        pass
    return os.environ.get(key, default)


# ── HTTP helpers ───────────────────────────────────────────────────────────

def _http_get_json(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.loads(resp.read())


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    """Force DataFrame into the standard schema: OHLCV, DatetimeIndex ascending."""
    if df is None or df.empty:
        return pd.DataFrame()
    expected = ["Open", "High", "Low", "Close", "Volume"]
    for c in expected:
        if c not in df.columns:
            df[c] = 0.0 if c == "Volume" else float("nan")
    df = df[expected]
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


# ── FMP ───────────────────────────────────────────────────────────────────

def _fmp_ticker(ticker: str) -> str:
    """yfinance-style → FMP-style symbol normalisation."""
    if ticker.endswith("-USD"):
        base = ticker[:-4].upper()
        if base in _CRYPTO_BASES:
            return f"{base}USD"
    return ticker


def _fetch_fmp(ticker: str, start: str, end: str) -> pd.DataFrame:
    """FMP /stable endpoint returns a flat JSON list of bars.

    Older /api/v3/historical-price-full/ requires legacy plan access and 403s
    on the newer 'stable' tier; this implementation targets the supported path.
    """
    key = get_secret("FMP_API_KEY")
    if not key:
        raise RuntimeError("FMP_API_KEY not configured")
    sym = _fmp_ticker(ticker)
    params = urllib.parse.urlencode({
        "symbol": sym, "from": start, "to": end, "apikey": key,
    })
    url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?{params}"
    data = _http_get_json(url)
    # /stable returns either a list of bars or an error dict
    if isinstance(data, dict):
        if "Error Message" in data:
            raise RuntimeError(f"FMP: {data['Error Message']}")
        rows = data.get("historical", [])
    else:
        rows = data or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    out = pd.DataFrame(index=df.index)
    out["Open"] = pd.to_numeric(df.get("open"), errors="coerce")
    out["High"] = pd.to_numeric(df.get("high"), errors="coerce")
    out["Low"]  = pd.to_numeric(df.get("low"), errors="coerce")
    # Prefer adjusted close when present (dividend/split adjusted), else raw close
    close_src = df["adjClose"] if "adjClose" in df.columns else df["close"]
    out["Close"] = pd.to_numeric(close_src, errors="coerce")
    if "volume" in df.columns:
        out["Volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    else:
        out["Volume"] = 0.0
    return out


# ── TwelveData ────────────────────────────────────────────────────────────

def _td_ticker(ticker: str) -> str:
    if ticker.endswith("-USD"):
        base = ticker[:-4].upper()
        if base in _CRYPTO_BASES:
            return f"{base}/USD"
    return ticker


def _fetch_twelvedata(ticker: str, start: str, end: str) -> pd.DataFrame:
    key = get_secret("TWELVEDATA_API_KEY")
    if not key:
        raise RuntimeError("TWELVEDATA_API_KEY not configured")
    sym = _td_ticker(ticker)
    params = urllib.parse.urlencode({
        "symbol": sym,
        "interval": "1day",
        "start_date": start,
        "end_date": end,
        "apikey": key,
        "outputsize": 5000,
        "format": "JSON",
    })
    url = f"https://api.twelvedata.com/time_series?{params}"
    data = _http_get_json(url)
    if isinstance(data, dict) and data.get("status") == "error":
        raise RuntimeError(f"TwelveData: {data.get('message','unknown error')}")
    values = data.get("values", []) if isinstance(data, dict) else []
    if not values:
        return pd.DataFrame()
    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    out = pd.DataFrame(index=df.index)
    out["Open"] = pd.to_numeric(df["open"], errors="coerce")
    out["High"] = pd.to_numeric(df["high"], errors="coerce")
    out["Low"]  = pd.to_numeric(df["low"], errors="coerce")
    out["Close"] = pd.to_numeric(df["close"], errors="coerce")
    if "volume" in df.columns:
        out["Volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    else:
        out["Volume"] = 0.0
    return out


# ── yfinance (last resort) ────────────────────────────────────────────────

def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(
        ticker, start=start, end=end,
        progress=False, auto_adjust=True,
    )
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep]


# ── Provider chain ────────────────────────────────────────────────────────

PROVIDERS: list[tuple[str, Callable[[str, str, str], pd.DataFrame]]] = [
    ("FMP",        _fetch_fmp),
    ("TwelveData", _fetch_twelvedata),
    ("yfinance",   _fetch_yfinance),
]


@st.cache_data(ttl=3600, show_spinner=False)
def get_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    """OHLCV daily data for `ticker` over [start, end] (inclusive end).

    Tries providers in order, returning the first non-empty result.
    Cached for 1 hour. Returns an empty DataFrame if every provider fails;
    callers should handle empty results gracefully.
    """
    for name, fn in PROVIDERS:
        try:
            df = fn(ticker, start, end)
            if df is not None and not df.empty:
                df = _normalize_history(df)
                # Stamp the source on the DataFrame for downstream debugging
                df.attrs["source"] = name
                df.attrs["ticker"] = ticker
                return df
        except Exception:
            continue
    return pd.DataFrame()


def get_history_recent(ticker: str, lookback_days: int) -> pd.DataFrame:
    """Convenience wrapper: fetch ~`lookback_days` worth of recent data.

    Computes start/end so calls within the same trading day share cache hits.
    """
    end = pd.Timestamp.today().normalize()
    # Buffer for weekends/holidays (use 1.6x for safety)
    start = end - pd.Timedelta(days=int(lookback_days * 1.6))
    return get_history(ticker, str(start.date()), str(end.date()))


def clear_cache() -> None:
    """Drop the cached price history. Call this from a dashboard's Refresh button."""
    try:
        get_history.clear()
    except Exception:
        pass


# ── Provider availability summary (handy for status banners) ──────────────

def available_providers() -> list[str]:
    """Names of providers with credentials currently configured (FMP, TwelveData)
    plus 'yfinance' which never needs a key."""
    out: list[str] = []
    if get_secret("FMP_API_KEY"):
        out.append("FMP")
    if get_secret("TWELVEDATA_API_KEY"):
        out.append("TwelveData")
    out.append("yfinance")
    return out
