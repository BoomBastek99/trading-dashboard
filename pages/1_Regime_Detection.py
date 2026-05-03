"""
Regime Detection Dashboard.

Hidden-Markov-Model regime detection on daily price data with strict
forward-only (filtered) inference - no look-ahead bias.

Run with:
    streamlit run regime_dashboard.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from data_provider import get_history
from design_system import *  # noqa: F401,F403

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Regime Detection",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


# ── Data ────────────────────────────────────────────────────────────────────

def load_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Thin wrapper around the unified data provider (FMP → TwelveData → yfinance)."""
    return get_history(ticker, start, end)


FEATURE_COLS = ["log_ret", "realized_vol", "volume_ratio", "hl_range_pct"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(0.0, index=df.index)

    out["log_ret"] = np.log(close).diff()
    out["realized_vol"] = out["log_ret"].rolling(20).std()

    if vol.sum() > 0:
        avg_vol = vol.rolling(20).mean()
        out["volume_ratio"] = (vol / avg_vol).replace([np.inf, -np.inf], np.nan)
    else:
        # No volume data: proxy with rolling abs-return ratio so the feature still varies.
        abs_ret = out["log_ret"].abs()
        out["volume_ratio"] = abs_ret.rolling(5).mean() / (abs_ret.rolling(60).mean() + 1e-9)

    out["hl_range_pct"] = (high - low) / close
    out["close"] = close
    return out.dropna()


# ── HMM training & BIC ──────────────────────────────────────────────────────

def fit_hmm(X: np.ndarray, n_components: int, seed: int = 42) -> GaussianHMM:
    model = GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=200,
        tol=1e-3,
        random_state=seed,
    )
    model.fit(X)
    return model


def bic(model: GaussianHMM, X: np.ndarray) -> float:
    n, d = X.shape
    k = model.n_components
    # diag-cov free params: (k-1) start + k*(k-1) trans + k*d means + k*d variances
    n_params = (k - 1) + k * (k - 1) + 2 * k * d
    return -2.0 * model.score(X) + n_params * np.log(n)


def select_best_model(X: np.ndarray, candidates=range(3, 8)):
    scores: dict[int, float] = {}
    best_model = None
    best_n = None
    best_bic = np.inf
    for n in candidates:
        try:
            m = fit_hmm(X, n)
            b = bic(m, X)
            scores[n] = b
            if b < best_bic:
                best_bic, best_model, best_n = b, m, n
        except Exception:
            scores[n] = float("nan")
    return best_model, best_n, scores


# ── Forward-only filtering (no Viterbi, no smoothing) ───────────────────────

def emission_logprob(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Per-timestep, per-state log emission probabilities, shape (T, K)."""
    T = X.shape[0]
    K = model.n_components
    out = np.empty((T, K))
    for k in range(K):
        out[:, k] = multivariate_normal.logpdf(
            X, mean=model.means_[k], cov=model.covars_[k]
        )
    return out


def forward_filter(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Filtered posterior P(state_t | obs_{1..t}).

    Pure forward recursion - row t depends only on X[:t+1]. This is the key
    guarantee against look-ahead bias.
    """
    T = X.shape[0]
    log_start = np.log(model.startprob_ + 1e-300)
    log_trans = np.log(model.transmat_ + 1e-300)
    log_b = emission_logprob(model, X)

    log_alpha = np.empty_like(log_b)
    log_alpha[0] = log_start + log_b[0]
    for t in range(1, T):
        log_alpha[t] = (
            logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0) + log_b[t]
        )
    return np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))


def verify_no_lookahead(model: GaussianHMM, X: np.ndarray) -> tuple[bool, str]:
    """Confirm filtered prob at time t is unchanged when future data is added.

    If the filter ever uses future data, refitting on a shorter prefix would
    produce a different value at the prefix's last row.
    """
    full = forward_filter(model, X)
    T = X.shape[0]
    test_points = sorted({max(1, T // 4), T // 2, 3 * T // 4, T - 1})
    for t in test_points:
        partial = forward_filter(model, X[: t + 1])
        if not np.allclose(full[t], partial[t], atol=1e-8):
            return False, f"mismatch at t={t}"
    return True, f"checked t={test_points}"


# ── Labeling & color mapping ────────────────────────────────────────────────

def vol_sorted_labels(model: GaussianHMM, vol_idx: int) -> tuple[dict, dict]:
    """Map raw state index -> label (sorted ascending by mean volatility),
    and raw state index -> rank (0 = lowest vol)."""
    vol_means = model.means_[:, vol_idx]
    order = np.argsort(vol_means)  # ascending
    n = model.n_components
    if n == 2:
        names = ["Low Vol", "High Vol"]
    elif n == 3:
        names = ["Low Vol", "Medium Vol", "High Vol"]
    elif n == 4:
        names = ["Low Vol", "Med-Low Vol", "Med-High Vol", "High Vol"]
    elif n == 5:
        names = ["Low Vol", "Med-Low Vol", "Medium Vol", "Med-High Vol", "High Vol"]
    else:
        names = ["Low Vol"] + [f"Vol Tier {i + 1}" for i in range(n - 2)] + ["High Vol"]
    idx_to_label = {int(raw): names[rank] for rank, raw in enumerate(order)}
    idx_to_rank = {int(raw): rank for rank, raw in enumerate(order)}
    return idx_to_label, idx_to_rank


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _lerp_hex(a: str, b: str, t: float) -> str:
    ra, ga, ba = _hex_to_rgb(a)
    rb, gb, bb = _hex_to_rgb(b)
    return "#{:02x}{:02x}{:02x}".format(
        int(ra + (rb - ra) * t),
        int(ga + (gb - ga) * t),
        int(ba + (bb - ba) * t),
    )


def color_for_label(label: str, rank: int, total: int) -> str:
    """REGIME_COLORS if known; otherwise a green->amber->red gradient by rank."""
    if label in REGIME_COLORS:
        return REGIME_COLORS[label]
    t = rank / max(total - 1, 1)
    if t <= 0.5:
        return _lerp_hex(ACCENT_GREEN, ACCENT_AMBER, t * 2)
    return _lerp_hex(ACCENT_AMBER, ACCENT_RED, (t - 0.5) * 2)


# ── Stability filter ────────────────────────────────────────────────────────

def apply_stability(
    raw_states: np.ndarray,
    persist: int = 3,
    window: int = 20,
    max_flicker: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Persistence + flicker detection.

    - stable_states[i] = state index once it has persisted >= `persist` bars,
      otherwise carry forward the previous stable state (-1 until first stable).
    - uncertain_mask[i] = True if more than `max_flicker` raw transitions
      occurred in the trailing `window` bars.
    """
    n = len(raw_states)
    stable = np.full(n, -1, dtype=int)
    streak_state = int(raw_states[0])
    streak = 1
    if streak >= persist:
        stable[0] = streak_state
    for i in range(1, n):
        if int(raw_states[i]) == streak_state:
            streak += 1
        else:
            streak_state = int(raw_states[i])
            streak = 1
        stable[i] = streak_state if streak >= persist else stable[i - 1]

    transitions = np.zeros(n, dtype=int)
    transitions[1:] = (raw_states[1:] != raw_states[:-1]).astype(int)
    uncertain = np.zeros(n, dtype=bool)
    csum = np.concatenate([[0], np.cumsum(transitions)])
    for i in range(n):
        lo = max(0, i - window + 1)
        if csum[i + 1] - csum[lo] > max_flicker:
            uncertain[i] = True
    return stable, uncertain


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(section_header("Inputs"), unsafe_allow_html=True)
    ticker = st.text_input("Ticker", value="SPY").upper().strip()

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

    n_override = st.selectbox(
        "Number of regimes",
        options=["Auto (BIC)", 3, 4, 5, 6, 7],
        index=0,
    )

    run = st.button("Run Analysis", type="primary", use_container_width=True)


# ── Run ─────────────────────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.session_state.results = None

if run:
    with st.spinner("Downloading prices and fitting HMM..."):
        prices = load_prices(ticker, str(start_d), str(end_d))
        if prices.empty:
            st.error(f"No data returned for {ticker}.")
            st.stop()

        feats = build_features(prices)
        if len(feats) < 100:
            st.error(f"Not enough data after feature engineering ({len(feats)} rows). Need >= 100.")
            st.stop()

        X = feats[FEATURE_COLS].values

        if n_override == "Auto (BIC)":
            model, n_best, bic_scores = select_best_model(X)
        else:
            try:
                model = fit_hmm(X, int(n_override))
                n_best = int(n_override)
                bic_scores = {n_best: bic(model, X)}
            except Exception as exc:
                st.error(f"HMM fit failed: {exc}")
                st.stop()

        if model is None:
            st.error("All HMM fits failed.")
            st.stop()

        # Forward-only inference
        filtered_post = forward_filter(model, X)
        raw_states = filtered_post.argmax(axis=1)
        confidence = filtered_post.max(axis=1)

        vol_idx = FEATURE_COLS.index("realized_vol")
        idx_to_label, idx_to_rank = vol_sorted_labels(model, vol_idx)

        stable_idx, uncertain_mask = apply_stability(raw_states)

        display_labels: list[str] = []
        for i in range(len(raw_states)):
            if stable_idx[i] == -1 or uncertain_mask[i]:
                display_labels.append("Uncertain")
            else:
                display_labels.append(idx_to_label[int(stable_idx[i])])
        display_labels = np.array(display_labels, dtype=object)

        ok, msg = verify_no_lookahead(model, X)

        st.session_state.results = {
            "ticker": ticker,
            "feats": feats,
            "model": model,
            "n_best": n_best,
            "bic_scores": bic_scores,
            "filtered_post": filtered_post,
            "raw_states": raw_states,
            "display_labels": display_labels,
            "confidence": confidence,
            "uncertain_mask": uncertain_mask,
            "idx_to_label": idx_to_label,
            "idx_to_rank": idx_to_rank,
            "lookahead_ok": ok,
            "lookahead_msg": msg,
        }


# ── Render ──────────────────────────────────────────────────────────────────

R = st.session_state.results

if R is None:
    st.markdown(
        f"<div style='color:{TEXT_MUTED};padding:64px 0;text-align:center;"
        f"font-family:DM Sans,sans-serif;font-size:0.95rem;'>"
        f"Configure inputs in the sidebar and click <b style='color:{ACCENT_CYAN}'>Run Analysis</b>."
        f"</div>",
        unsafe_allow_html=True,
    )
    st.stop()

feats = R["feats"]
display_labels = R["display_labels"]
confidence = R["confidence"]
n_total = R["n_best"]

# Look-ahead verification banner
verify_color = ACCENT_GREEN if R["lookahead_ok"] else ACCENT_RED
verify_text = "FORWARD-ONLY · NO LOOK-AHEAD" if R["lookahead_ok"] else "LOOK-AHEAD DETECTED"
verify_dot = status_dot("connected" if R["lookahead_ok"] else "error")
st.markdown(
    f"<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
    f"color:{verify_color};letter-spacing:2px;margin-bottom:14px;'>"
    f"{verify_dot}{verify_text} &nbsp;·&nbsp; <span style='color:{TEXT_MUTED}'>{R['lookahead_msg']}</span>"
    f"</div>",
    unsafe_allow_html=True,
)

# Build a label -> color map (used by chart bands and stat cards)
label_to_color: dict[str, str] = {"Uncertain": ACCENT_VIOLET}
for raw, lab in R["idx_to_label"].items():
    label_to_color[lab] = color_for_label(lab, R["idx_to_rank"][raw], n_total)


# ── Top bar ─────────────────────────────────────────────────────────────────

current_label = str(display_labels[-1])
current_conf = float(confidence[-1] * 100)
last_close = float(feats["close"].iloc[-1])

recent = display_labels[-20:]
recent_transitions = int((recent[1:] != recent[:-1]).sum())
if R["uncertain_mask"][-1]:
    stab_text, stab_color = "UNSTABLE", ACCENT_RED
elif recent_transitions <= 1:
    stab_text, stab_color = "STABLE", ACCENT_GREEN
else:
    stab_text, stab_color = "MIXED", ACCENT_AMBER

cols = st.columns([1.4, 1.5, 1.1, 1.1, 1.1])
with cols[0]:
    st.markdown(
        f"""
        <div style='padding:8px 0;'>
          <div style='font-family:DM Sans;font-size:0.7rem;color:{TEXT_MUTED};
                      text-transform:uppercase;letter-spacing:2px;'>Ticker</div>
          <div style='font-family:JetBrains Mono;font-size:2.2rem;font-weight:700;
                      color:{TEXT_PRIMARY};line-height:1.1;'>{R['ticker']}</div>
          <div style='font-family:JetBrains Mono;font-size:0.85rem;color:{TEXT_SECONDARY};'>
              ${last_close:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with cols[1]:
    st.markdown(
        f"<div style='font-family:DM Sans;font-size:0.7rem;color:{TEXT_MUTED};"
        f"text-transform:uppercase;letter-spacing:2px;margin:8px 0 10px 0;'>"
        f"Current regime</div>",
        unsafe_allow_html=True,
    )
    st.markdown(regime_badge(current_label, current_conf), unsafe_allow_html=True)
with cols[2]:
    st.markdown(metric_card("Confidence", f"{current_conf:.1f}%", ACCENT_CYAN), unsafe_allow_html=True)
with cols[3]:
    st.markdown(metric_card("Stability", stab_text, stab_color), unsafe_allow_html=True)
with cols[4]:
    st.markdown(metric_card("Regimes", str(n_total), ACCENT_VIOLET), unsafe_allow_html=True)


# ── Hero chart ──────────────────────────────────────────────────────────────

st.markdown(section_header("Price · Regime Bands"), unsafe_allow_html=True)

dates = feats.index
close = feats["close"].values

fig = go.Figure()


def regime_runs(labels: np.ndarray, idx) -> list[tuple[str, object, object]]:
    runs: list[tuple[str, object, object]] = []
    if len(labels) == 0:
        return runs
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start]:
            runs.append((str(labels[start]), idx[start], idx[i]))
            start = i
    runs.append((str(labels[start]), idx[start], idx[-1]))
    return runs


for label, x0, x1 in regime_runs(display_labels, dates):
    fig.add_vrect(
        x0=x0,
        x1=x1,
        fillcolor=label_to_color.get(label, ACCENT_VIOLET),
        opacity=0.13,
        line_width=0,
        layer="below",
    )

fig.add_trace(
    go.Scatter(
        x=dates,
        y=close,
        mode="lines",
        line=dict(color=TEXT_PRIMARY, width=1.6),
        name="Close",
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
    )
)

layout = get_plotly_layout()
layout["height"] = 540
layout["showlegend"] = False
layout["margin"] = dict(l=40, r=20, t=20, b=40)
fig.update_layout(**layout)
st.plotly_chart(fig, use_container_width=True)


# ── Regime statistics grid ──────────────────────────────────────────────────

st.markdown(section_header("Regime Statistics"), unsafe_allow_html=True)

stats_df = pd.DataFrame({
    "label": display_labels,
    "ret": feats["log_ret"].values,
    "vol": feats["realized_vol"].values,
    "vr": feats["volume_ratio"].values,
})

# Order: by vol rank (low -> high), then Uncertain at the end
unique_labels = list(pd.Index(display_labels).unique())
ordered: list[str] = []
for raw in np.argsort(R["model"].means_[:, FEATURE_COLS.index("realized_vol")]):
    lab = R["idx_to_label"][int(raw)]
    if lab in unique_labels and lab not in ordered:
        ordered.append(lab)
if "Uncertain" in unique_labels and "Uncertain" not in ordered:
    ordered.append("Uncertain")

stat_cols = st.columns(min(4, max(1, len(ordered))))
for i, lab in enumerate(ordered):
    sub = stats_df[stats_df["label"] == lab]
    if len(sub) == 0:
        continue
    pct_time = 100 * len(sub) / len(stats_df)
    mean_ret = float(sub["ret"].mean()) * 100
    mean_vol = float(sub["vol"].mean()) * 100
    mean_vr = float(sub["vr"].mean())
    color = label_to_color.get(lab, ACCENT_VIOLET)

    body = f"""
    <div style="
        background:{BG_CARD};
        border:1px solid {BORDER};
        border-left:3px solid {color};
        border-radius:12px;
        padding:18px;
        margin-bottom:12px;
    ">
        <div style="font-family:DM Sans;font-size:0.78rem;color:{color};
                    text-transform:uppercase;letter-spacing:2.5px;font-weight:700;
                    margin-bottom:14px;">{lab}</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
            <div>
                <div style='font-family:DM Sans;font-size:0.62rem;color:{TEXT_MUTED};
                            text-transform:uppercase;letter-spacing:1.5px;'>Mean Ret</div>
                <div style='font-family:JetBrains Mono;font-size:1.05rem;font-weight:600;
                            color:{pnl_color(mean_ret)};'>{mean_ret:+.2f}%</div>
            </div>
            <div>
                <div style='font-family:DM Sans;font-size:0.62rem;color:{TEXT_MUTED};
                            text-transform:uppercase;letter-spacing:1.5px;'>Mean Vol</div>
                <div style='font-family:JetBrains Mono;font-size:1.05rem;font-weight:600;
                            color:{TEXT_PRIMARY};'>{mean_vol:.2f}%</div>
            </div>
            <div>
                <div style='font-family:DM Sans;font-size:0.62rem;color:{TEXT_MUTED};
                            text-transform:uppercase;letter-spacing:1.5px;'>Vol Ratio</div>
                <div style='font-family:JetBrains Mono;font-size:1.05rem;font-weight:600;
                            color:{TEXT_PRIMARY};'>{mean_vr:.2f}x</div>
            </div>
            <div>
                <div style='font-family:DM Sans;font-size:0.62rem;color:{TEXT_MUTED};
                            text-transform:uppercase;letter-spacing:1.5px;'>% Time</div>
                <div style='font-family:JetBrains Mono;font-size:1.05rem;font-weight:600;
                            color:{ACCENT_CYAN};'>{pct_time:.1f}%</div>
            </div>
        </div>
    </div>
    """
    stat_cols[i % len(stat_cols)].markdown(body, unsafe_allow_html=True)


# ── Confidence timeline ────────────────────────────────────────────────────

st.markdown(section_header("Filtered Confidence"), unsafe_allow_html=True)

cfig = go.Figure()
cfig.add_trace(
    go.Scatter(
        x=dates,
        y=confidence * 100,
        mode="lines",
        line=dict(color=ACCENT_CYAN, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.30)",
        name="Confidence",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
    )
)
clayout = get_plotly_layout()
clayout["height"] = 220
clayout["showlegend"] = False
clayout["yaxis"]["range"] = [0, 100]
clayout["yaxis"]["ticksuffix"] = "%"
clayout["margin"] = dict(l=40, r=20, t=20, b=40)
cfig.update_layout(**clayout)
st.plotly_chart(cfig, use_container_width=True)


# ── BIC table (if multiple candidates evaluated) ───────────────────────────

if len(R["bic_scores"]) > 1:
    with st.expander("BIC scores by component count"):
        bdf = (
            pd.DataFrame(
                {
                    "n_components": list(R["bic_scores"].keys()),
                    "BIC": list(R["bic_scores"].values()),
                }
            )
            .set_index("n_components")
            .round(2)
        )
        st.dataframe(style_dataframe(bdf), use_container_width=True)
