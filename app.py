"""Machine Spirits — Interactive Dashboard.

Companion to "Machine Spirits: Speculation and Adaptation of LLM Agents in
Asset Markets". Three pages:

  1. A taxonomy of Machine Spirits — homogeneous single-LLM markets.
  2. Mixed Market Chaos            — seed-driven divergence in a 6-model mix.
  3. The Adaptation Duel           — Gemini vs GPT-5-mini vs Qwen baseline.

Run:   streamlit run app.py
Data:  dashboard_data.parquet + runs_meta.parquet (produced by prepare_data.py).
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_FILE = Path(__file__).parent / "dashboard_data.parquet"
META_FILE = Path(__file__).parent / "runs_meta.parquet"

FUNDAMENTAL_PRICE = 60.0
VOLATILITY_THRESHOLD = 100.0

CHAOS_GROUP = (
    "1Qwen-Qwen3-14B_and_1allenai-Olmo-3-7B-Think"
    "_and_1deepseek-ai-DeepSeek-R1-Distill-Llama-8B"
    "_and_1gemini-3-flash-preview_and_1gemma-3-27b-it_and_1gpt-5-mini"
)
QWEN_GROUP   = "6Qwen-Qwen3-14B"
GEMINI_GROUP = "5Qwen-Qwen3-14B_and_1gemini-3-flash-preview"
GPT5_GROUP   = "5Qwen-Qwen3-14B_and_1gpt-5-mini"

QWEN_BASELINE_RUN_ID = 46
GEMINI_HERO_RUN_ID   = 295
GPT5_HERO_RUN_ID     = 7

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "6Qwen-Qwen3-14B":                           "Qwen-3 14B (×6)",
    "6Qwen-Qwen3-14B_no_reasoning":              "Qwen-3 14B, no reasoning (×6)",
    "6Qwen-Qwen3-32B":                           "Qwen-3 32B (×6)",
    "6Qwen-Qwen2-5-7B-Instruct":                 "Qwen-2.5 7B Instruct (×6)",
    "6gemini-2-5-flash":                         "Gemini-2.5 Flash (×6)",
    "6gemini-3-flash-preview":                   "Gemini-3 Flash preview (×6)",
    "6gemma-3-27b-it":                           "Gemma-3 27B Instruct (×6)",
    "6gpt-4-1":                                  "GPT-4.1 (×6)",
    "6gpt-4o-mini":                              "GPT-4o mini (×6)",
    "6gpt-5-mini":                               "GPT-5 mini (×6)",
    "6o3":                                       "OpenAI o3 (×6)",
    "6o3-mini":                                  "OpenAI o3-mini (×6)",
    "6allenai-Olmo-3-7B-Instruct":               "OLMo-3 7B Instruct (×6)",
    "6allenai-Olmo-3-7B-Think":                  "OLMo-3 7B Think (×6)",
    "6deepseek-ai-DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1 Distill Llama 8B (×6)",
    "6deepseek-ai-DeepSeek-R1-Distill-Qwen-14B": "DeepSeek-R1 Distill Qwen 14B (×6)",
    QWEN_GROUP:   "Qwen-3 14B (×6)",
    GEMINI_GROUP: "5× Qwen-3 14B + 1× Gemini-3 Flash",
    GPT5_GROUP:   "5× Qwen-3 14B + 1× GPT-5 mini",
    CHAOS_GROUP:  "Mixed (Qwen + OLMo + DeepSeek + Gemini + Gemma + GPT-5)",
}

# Plotly palette
ACTUAL_COLOR      = "#F2F2F2"
FUNDAMENTAL_COLOR = "#7CFFA1"
AGENT_PALETTE     = ["#5DA5DA", "#FAA43A", "#60BD68", "#F17CB0", "#B276B2", "#DECF3F"]
BG_COLOR          = "#0E1117"
GRID_COLOR        = "#1C2230"

GEMINI_COLOR      = "#FF5000"   # fire orange-red
GPT5_COLOR        = "#00B4FF"   # ice blue
QWEN_COLOR        = "#A0A8B8"   # neutral steel


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Machine Spirits",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background-color: #0E1117; }
      h1, h2, h3 { letter-spacing: -0.02em; }
      [data-testid="stMetricValue"] { font-family: "SF Mono", "Menlo", monospace; }
      [data-testid="stMetricLabel"] {
          text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.75rem;
      }
      .ticker-pill {
          display: inline-block; padding: 2px 10px; border-radius: 999px;
          background: #1C2230; color: #9BA4B5; font-family: "SF Mono", monospace;
          font-size: 0.8rem; letter-spacing: 0.05em;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading market data...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DATA_FILE.exists() or not META_FILE.exists():
        st.error(
            f"Data files missing. Run `python prepare_data.py` first to "
            f"generate `{DATA_FILE.name}` and `{META_FILE.name}`."
        )
        st.stop()
    return pd.read_parquet(DATA_FILE), pd.read_parquet(META_FILE)


def display_name(model_group: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_group, model_group)


def actual_series(prices: pd.DataFrame, run_id: int) -> pd.DataFrame:
    sub = prices[prices.run_id == run_id]
    anchor = sub.agent_id.min()
    return (sub[sub.agent_id == anchor]
            .sort_values("time_step")[["time_step", "actual_price"]]
            .reset_index(drop=True))


def predicted_pivot(prices: pd.DataFrame, run_id: int) -> pd.DataFrame:
    sub = prices[prices.run_id == run_id]
    return (sub.pivot(index="time_step", columns="agent_id", values="predicted_price")
            .sort_index())


# ---------------------------------------------------------------------------
# Playback controls (Streamlit-native, rendered ABOVE the chart)
# ---------------------------------------------------------------------------

def render_playback_controls(key_prefix: str) -> int:
    """Renders ▶ / ⏸ buttons + a Streamlit time-step slider above the chart.
    Returns the current time step t (0–49).

    Uses an unbound internal key (`{prefix}_t`) so that advance_playback can
    freely mutate it between reruns — Streamlit forbids mutating a key that is
    directly tied to a widget via `key=`.
    """
    play_key = f"{key_prefix}_playing"
    t_key    = f"{key_prefix}_t"   # internal; NOT the slider widget key

    if play_key not in st.session_state:
        st.session_state[play_key] = False
    if t_key not in st.session_state:
        st.session_state[t_key] = 0

    c_play, c_pause, c_slider = st.columns([1, 1, 10])
    with c_play:
        if st.button("▶", key=f"{key_prefix}_play_btn", help="Play from start"):
            st.session_state[play_key] = True
            st.session_state[t_key]    = 0
    with c_pause:
        if st.button("⏸", key=f"{key_prefix}_pause_btn", help="Pause"):
            st.session_state[play_key] = False
    with c_slider:
        # No key= here; value= is driven by our internal state.
        # If the user drags, the returned value differs → we detect and sync.
        dragged = st.slider(
            "Time step", 0, 49,
            value=st.session_state[t_key],
            label_visibility="collapsed",
        )
        if dragged != st.session_state[t_key]:
            st.session_state[t_key]    = dragged
            st.session_state[play_key] = False   # dragging stops auto-play

    return int(st.session_state[t_key])


def advance_playback(key_prefix: str, t: int) -> None:
    """If currently playing, sleep briefly, increment t by 1, and rerun."""
    play_key = f"{key_prefix}_playing"
    t_key    = f"{key_prefix}_t"
    if st.session_state.get(play_key, False):
        if t < 49:
            time.sleep(0.07)
            st.session_state[t_key] = t + 1   # safe: t_key is NOT widget-bound
            st.rerun()
        else:
            st.session_state[play_key] = False


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def base_layout(height: int = 480) -> dict:
    return dict(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color="#C8CDD7", family="Inter, system-ui, sans-serif"),
        margin=dict(l=50, r=20, t=20, b=90),
        height=height,
        xaxis=dict(
            title="Time step", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
            range=[0, 49],
        ),
        yaxis=dict(title="Price", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        legend=dict(
            orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )


def add_fundamental_line(fig: go.Figure) -> None:
    fig.add_hline(
        y=FUNDAMENTAL_PRICE,
        line_dash="dash",
        line_color=FUNDAMENTAL_COLOR,
        opacity=0.6,
        annotation_text=f"  p_f = {FUNDAMENTAL_PRICE:.0f}",
        annotation_position="top right",
        annotation=dict(font=dict(color=FUNDAMENTAL_COLOR, size=11)),
    )


def snapshot_figure(
    prices: pd.DataFrame,
    run_id: int,
    t: int,
    show_predictions: bool,
    height: int = 480,
) -> go.Figure:
    """Snapshot of a single run up to time step t (for Pages 1 & 2)."""
    actual = actual_series(prices, run_id)
    actual_t = actual[actual.time_step <= t]

    y_max = float(actual.actual_price.max())

    fig = go.Figure(layout=base_layout(height=height))

    if show_predictions:
        pred = predicted_pivot(prices, run_id)
        y_max = max(y_max, float(pred.max().max()))
        pred_t = pred[pred.index <= t]
        for i, agent_id in enumerate(pred_t.columns):
            fig.add_trace(go.Scatter(
                x=pred_t.index, y=pred_t[agent_id],
                mode="lines",
                name=f"Agent {agent_id} forecast",
                line=dict(color=AGENT_PALETTE[i % len(AGENT_PALETTE)], width=1),
                opacity=0.55,
            ))

    fig.add_trace(go.Scatter(
        x=actual_t.time_step, y=actual_t.actual_price,
        mode="lines",
        name="Actual price",
        line=dict(color=ACTUAL_COLOR, width=3),
    ))

    fig.update_layout(yaxis_range=[-5, y_max * 1.06 + 5])
    add_fundamental_line(fig)
    return fig


def duel_snapshot_figure(prices: pd.DataFrame, t: int, height: int = 560) -> go.Figure:
    """Single-panel overlay of all three duel runs up to time step t.

    Gemini gets a fire glow (orange-red), GPT-5 gets an ice glow (blue),
    Qwen stays neutral steel. All three lines reveal together as t advances.
    """
    # Fix y-axis to full data range so it never jumps during playback
    y_max = 0.0
    for rid in (QWEN_BASELINE_RUN_ID, GEMINI_HERO_RUN_ID, GPT5_HERO_RUN_ID):
        y_max = max(y_max, float(actual_series(prices, rid).actual_price.max()))

    fig = go.Figure(layout=base_layout(height=height))
    fig.update_layout(yaxis_range=[-5, y_max * 1.06 + 5])

    # ── Qwen baseline — neutral steel ────────────────────────────────────────
    qwen = actual_series(prices, QWEN_BASELINE_RUN_ID)
    qwen_t = qwen[qwen.time_step <= t]
    fig.add_trace(go.Scatter(
        x=qwen_t.time_step, y=qwen_t.actual_price,
        mode="lines",
        name="🧭  Qwen-3 14B — Baseline",
        line=dict(color=QWEN_COLOR, width=2.5, dash="dot"),
    ))

    # ── Gemini — fire glow ────────────────────────────────────────────────────
    gem = actual_series(prices, GEMINI_HERO_RUN_ID)
    gem_t = gem[gem.time_step <= t]
    # Wide semi-transparent halo for the glow effect
    fig.add_trace(go.Scatter(
        x=gem_t.time_step, y=gem_t.actual_price,
        mode="lines", showlegend=False, hoverinfo="skip",
        line=dict(color=f"rgba(255,80,0,0.15)", width=16),
    ))
    fig.add_trace(go.Scatter(
        x=gem_t.time_step, y=gem_t.actual_price,
        mode="lines",
        name="🔥  Gemini-3 Flash — Apex Predator",
        line=dict(color=GEMINI_COLOR, width=3),
    ))

    # ── GPT-5 mini — ice glow ────────────────────────────────────────────────
    gpt = actual_series(prices, GPT5_HERO_RUN_ID)
    gpt_t = gpt[gpt.time_step <= t]
    fig.add_trace(go.Scatter(
        x=gpt_t.time_step, y=gpt_t.actual_price,
        mode="lines", showlegend=False, hoverinfo="skip",
        line=dict(color=f"rgba(0,180,255,0.15)", width=16),
    ))
    fig.add_trace(go.Scatter(
        x=gpt_t.time_step, y=gpt_t.actual_price,
        mode="lines",
        name="🧊  GPT-5 mini — Volatility Damper",
        line=dict(color=GPT5_COLOR, width=3),
    ))

    add_fundamental_line(fig)
    return fig


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

def render_sidebar() -> str:
    st.sidebar.title("Machine Spirits")
    st.sidebar.caption("Speculation & adaptation of LLM agents in asset markets")
    return st.sidebar.radio(
        "Pages",
        ["1 — A taxonomy of Machine Spirits",
         "2 — Mixed Market Chaos",
         "3 — The Adaptation Duel"],
        label_visibility="collapsed",
    )


# ---------------------------------------------------------------------------
# Page 1: A taxonomy of Machine Spirits
# ---------------------------------------------------------------------------

def page_spirit_gallery(prices: pd.DataFrame, meta: pd.DataFrame) -> None:
    st.title("A taxonomy of Machine Spirits")
    st.caption(
        "Each LLM has its own economic disposition. Pick a homogeneous market "
        "(six copies of the same model) and watch how the price evolves."
    )

    homogeneous = sorted(
        meta[~meta.model_group.str.contains("_and_") &
             meta.model_group.str.startswith("6")]
        .model_group.unique()
    )

    chosen_group = st.selectbox(
        "Market composition",
        options=homogeneous,
        format_func=display_name,
        index=homogeneous.index(QWEN_GROUP) if QWEN_GROUP in homogeneous else 0,
    )

    # Pick a new random run whenever the model changes; reset playback.
    if st.session_state.get("p1_group") != chosen_group:
        st.session_state["p1_group"]   = chosen_group
        group_runs = meta[meta.model_group == chosen_group]
        st.session_state["p1_run_id"]  = int(random.choice(group_runs.run_id.tolist()))
        st.session_state["p1_t"]       = 0
        st.session_state["p1_playing"] = False

    chosen_run = st.session_state["p1_run_id"]

    show_pred = st.checkbox(
        "Show agent forecasts", value=True,
        help="Overlay each of the 6 agents' next-period price expectations.",
    )

    t = render_playback_controls("p1")
    fig = snapshot_figure(prices, chosen_run, t, show_predictions=show_pred)
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "Bold line: the realized market price. Thin lines: each agent's "
        f"submitted forecast for the next period. The dashed line marks the "
        f"fundamental value p_f = {FUNDAMENTAL_PRICE:.0f}."
    )

    advance_playback("p1", t)


# ---------------------------------------------------------------------------
# Page 2: Mixed Market Chaos
# ---------------------------------------------------------------------------

def volatility_label(early_std: float, late_std: float) -> tuple[str, str]:
    early_hi = early_std > VOLATILITY_THRESHOLD
    late_hi  = late_std  > VOLATILITY_THRESHOLD
    if early_hi and late_hi:
        return "Persistent volatility", "error"
    if early_hi:
        return "Early volatility", "warning"
    if late_hi:
        return "Late volatility", "warning"
    return "Stable", "success"


def page_chaos(prices: pd.DataFrame, meta: pd.DataFrame) -> None:
    st.title("Mixed Market Chaos")
    st.caption(
        "Six different LLMs share a market. Same prompt, same parameters, "
        "same six models — only the random seed changes. Roll the dice and "
        "watch identical configurations produce wildly different macro outcomes."
    )

    chaos_runs = meta[meta.model_group == CHAOS_GROUP].sort_values("seed")
    if chaos_runs.empty:
        st.error(f"No runs found for model_group `{CHAOS_GROUP}`.")
        return

    if "chaos_run_id" not in st.session_state:
        st.session_state["chaos_run_id"] = int(chaos_runs.iloc[0].run_id)

    col_btn, col_caption = st.columns([1, 4])
    with col_btn:
        if st.button("🎲  Run Chaos Roulette", type="primary", use_container_width=True):
            st.session_state["chaos_run_id"] = int(random.choice(chaos_runs.run_id.tolist()))
            # Reset playback and auto-play so the new run animates from the start
            st.session_state["p2_t"]       = 0
            st.session_state["p2_playing"] = True
    with col_caption:
        st.markdown(
            f"<span class='ticker-pill'>{len(chaos_runs)} seeds available</span>",
            unsafe_allow_html=True,
        )

    run_id = st.session_state["chaos_run_id"]
    row    = chaos_runs[chaos_runs.run_id == run_id].iloc[0]

    show_pred = st.checkbox("Show agent forecasts", value=True, key="chaos_show_pred")

    st.markdown(
        f"<div style='margin: 4px 0 -8px 4px; color:#9BA4B5;'>"
        f"<span class='ticker-pill'>Seed {int(row.seed)}</span></div>",
        unsafe_allow_html=True,
    )

    t = render_playback_controls("p2")
    fig = snapshot_figure(prices, int(run_id), t, show_predictions=show_pred, height=480)
    st.plotly_chart(fig, width="stretch")

    label, severity = volatility_label(row.early_std, row.late_std)
    m1, m2, m3 = st.columns(3)
    m1.metric("Early std (t < 25)", f"{row.early_std:.1f}",
              delta="volatile" if row.early_std > VOLATILITY_THRESHOLD else "calm",
              delta_color="off")
    m2.metric("Late std (t ≥ 25)", f"{row.late_std:.1f}",
              delta="volatile" if row.late_std  > VOLATILITY_THRESHOLD else "calm",
              delta_color="off")
    m3.metric("Peak price", f"{row.peak_price:.1f}")

    {"success": st.success, "warning": st.warning, "error": st.error}[severity](
        f"**Regime:** {label}  ·  threshold = std > {VOLATILITY_THRESHOLD:.0f}"
    )

    advance_playback("p2", t)


# ---------------------------------------------------------------------------
# Page 3: The Adaptation Duel
# ---------------------------------------------------------------------------

def page_adaptation_duel(prices: pd.DataFrame, meta: pd.DataFrame) -> None:
    st.title("The Adaptation Duel")
    st.caption(
        "Five Qwen-3 14B trend-followers, plus one adaptive agent. Swap that "
        "single agent and the market changes character entirely. Hit ▶ to watch "
        "all three markets unfold together — fire, ice, and the neutral baseline."
    )

    t = render_playback_controls("p3")
    fig = duel_snapshot_figure(prices, t)
    st.plotly_chart(fig, width="stretch")

    advance_playback("p3", t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    prices, meta = load_data()
    page = render_sidebar()
    if page.startswith("1"):
        page_spirit_gallery(prices, meta)
    elif page.startswith("2"):
        page_chaos(prices, meta)
    else:
        page_adaptation_duel(prices, meta)


if __name__ == "__main__":
    main()
