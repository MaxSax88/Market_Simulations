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

# Page 1: fixed default run per model group (non-random)
DEFAULT_RUN_IDS: dict[str, int] = {
    "6Qwen-Qwen2-5-7B-Instruct":                 368,
    "6Qwen-Qwen3-14B":                            46,
    "6Qwen-Qwen3-14B_no_reasoning":               109,  
    "6Qwen-Qwen3-32B":                            16,
    "6allenai-Olmo-3-7B-Instruct":                517,  
    "6allenai-Olmo-3-7B-Think":                   276,
    "6deepseek-ai-DeepSeek-R1-Distill-Llama-8B":  202,
    "6deepseek-ai-DeepSeek-R1-Distill-Qwen-14B":  238,
    "6gemini-2-5-flash":                          50,
    "6gemini-3-flash-preview":                    186,
    "6gemma-3-27b-it":                            340,
    "6gpt-4-1":                                   412,
    "6gpt-4o-mini":                               284,
    "6gpt-5-mini":                                362,
    "6o3":                                        434,
    "6o3-mini":                                   404,
}

# Page 2: run classification thresholds and category labels/colours
BUBBLE_PRICE_THRESHOLD      = 300
BUBBLE_CONSEC_STEPS         = 3    # consecutive steps above threshold to count as bubble
BUBBLE_STD_THRESHOLD        = 100
NO_BUBBLE_STD_THRESHOLD     = 20

CAT_LABELS = [
    "No bubble\n(low volatility)",
    "No bubble\n(volatility)",
    "Bubble\n(early volatility)",
    "Bubble\n(late volatility)",
    "Bubble\n(persistent volatility)",
]
CAT_COLORS = ["#60BD68", "#DECF3F", "#FAA43A", "#E05530", "#B03020"]


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
    meta   = pd.read_parquet(META_FILE)
    prices = pd.read_parquet(DATA_FILE)
    # Keep only standard runs (res_seed_<int>.pkl); exclude extrapolate/nonlinear variants
    meta   = meta[meta["filename"].str.match(r"^res_seed_\d+\.pkl$", na=False)]
    prices = prices[prices["run_id"].isin(meta["run_id"])]
    return prices, meta


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


def has_bubble(price_series: "pd.Series") -> bool:
    """True if price exceeds BUBBLE_PRICE_THRESHOLD for BUBBLE_CONSEC_STEPS consecutive steps."""
    above = (price_series > BUBBLE_PRICE_THRESHOLD).values
    consec = 0
    for v in above:
        consec = consec + 1 if v else 0
        if consec >= BUBBLE_CONSEC_STEPS:
            return True
    return False


def classify_run(bubble: bool, early_std: float, late_std: float) -> str:
    if not bubble:
        if early_std < NO_BUBBLE_STD_THRESHOLD and late_std < NO_BUBBLE_STD_THRESHOLD:
            return CAT_LABELS[0]
        return CAT_LABELS[1]
    if early_std >= BUBBLE_STD_THRESHOLD and late_std >= BUBBLE_STD_THRESHOLD:
        return CAT_LABELS[4]
    if early_std >= BUBBLE_STD_THRESHOLD:
        return CAT_LABELS[2]
    if late_std >= BUBBLE_STD_THRESHOLD:
        return CAT_LABELS[3]
    return CAT_LABELS[4]


@st.cache_data(show_spinner=False)
def chaos_category_map(prices: pd.DataFrame, meta: pd.DataFrame) -> dict[int, str]:
    rows = meta[meta.model_group == CHAOS_GROUP]
    result: dict[int, str] = {}
    for _, r in rows.iterrows():
        run_id = int(r.run_id)
        price_series = actual_series(prices, run_id)["actual_price"]
        result[run_id] = classify_run(has_bubble(price_series), r.early_std, r.late_std)
    return result


def render_category_bar_chart(counts: dict[str, int]) -> go.Figure:
    total = sum(counts.values())
    # Sort highest → lowest so the tallest bar is always on the left
    order = sorted(range(len(CAT_LABELS)), key=lambda i: counts.get(CAT_LABELS[i], 0), reverse=True)
    labels = [CAT_LABELS[i] for i in order]
    colors = [CAT_COLORS[i] for i in order]
    vals   = [counts[l] for l in labels]
    texts  = [f"{v / total * 100:.0f}%" if total > 0 else "" for v in vals]
    fig    = go.Figure(go.Bar(
        x=labels, y=vals,
        marker_color=colors,
        text=texts,
        textposition="auto",
    ))
    title = f"Outcomes after {total} roll{'s' if total != 1 else ''}" if total > 0 else "Roll to begin"
    layout = base_layout(height=260)
    layout.update(
        xaxis=dict(title=None, tickfont=dict(size=11), gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(title="Count", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        margin=dict(l=50, r=20, t=50, b=80),
        title_text=title,
        title_font=dict(size=13, color="#C8CDD7"),
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_layout(layout)
    return fig


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
            time.sleep(0.022)
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

    fig = go.Figure(layout=base_layout(height=height))

    if show_predictions:
        pred   = predicted_pivot(prices, run_id)
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

    run_max = float(actual.actual_price.max())
    if run_max > 150:
        fig.update_layout(yaxis_range=[0, 1000])
    else:
        fig.update_layout(yaxis_range=[0, 150])
        fig.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.97,
            text="⚠ Axis capped at 150",
            showarrow=False,
            font=dict(size=13, color="#DECF3F"),
            align="left",
            bgcolor="rgba(14,17,23,0.82)",
            bordercolor="#DECF3F",
            borderwidth=1,
            borderpad=5,
        )

    add_fundamental_line(fig)
    return fig


def duel_snapshot_figure(prices: pd.DataFrame, t: int, height: int = 560) -> go.Figure:
    """Single-panel overlay of all three duel runs up to time step t.

    Gemini gets a fire glow (orange-red), GPT-5 gets an ice glow (blue),
    Qwen stays neutral steel. All three lines reveal together as t advances.
    """
    fig = go.Figure(layout=base_layout(height=height))

    # ── Qwen baseline — neutral steel ────────────────────────────────────────
    qwen   = actual_series(prices, QWEN_BASELINE_RUN_ID)
    qwen_t = qwen[qwen.time_step <= t]
    fig.add_trace(go.Scatter(
        x=qwen_t.time_step, y=qwen_t.actual_price,
        mode="lines",
        name="🧭  6x Qwen-3 14B — Baseline",
        line=dict(color=QWEN_COLOR, width=2.5, dash="dot"),
    ))

    # ── Gemini — fire glow ────────────────────────────────────────────────────
    gem   = actual_series(prices, GEMINI_HERO_RUN_ID)
    gem_t = gem[gem.time_step <= t]
    fig.add_trace(go.Scatter(
        x=gem_t.time_step, y=gem_t.actual_price,
        mode="lines", showlegend=False, hoverinfo="skip",
        line=dict(color="rgba(255,80,0,0.15)", width=16),
    ))
    fig.add_trace(go.Scatter(
        x=gem_t.time_step, y=gem_t.actual_price,
        mode="lines",
        name="🔥  1x Gemini-3 Flash vs 5x Qwen-3 14B",
        line=dict(color=GEMINI_COLOR, width=3),
    ))

    # ── GPT-5 mini — ice glow ────────────────────────────────────────────────
    gpt   = actual_series(prices, GPT5_HERO_RUN_ID)
    gpt_t = gpt[gpt.time_step <= t]
    fig.add_trace(go.Scatter(
        x=gpt_t.time_step, y=gpt_t.actual_price,
        mode="lines", showlegend=False, hoverinfo="skip",
        line=dict(color="rgba(0,180,255,0.15)", width=16),
    ))
    fig.add_trace(go.Scatter(
        x=gpt_t.time_step, y=gpt_t.actual_price,
        mode="lines",
        name="🧊  1x GPT-5 mini vs 5x Qwen-3 14B",
        line=dict(color=GPT5_COLOR, width=3),
    ))

    # Static y-axis: based on the full peak across all three runs
    run_max = max(
        float(actual_series(prices, QWEN_BASELINE_RUN_ID).actual_price.max()),
        float(actual_series(prices, GEMINI_HERO_RUN_ID).actual_price.max()),
        float(actual_series(prices, GPT5_HERO_RUN_ID).actual_price.max()),
    )
    if run_max > 150:
        fig.update_layout(yaxis_range=[0, 1000])
    else:
        fig.update_layout(yaxis_range=[0, 150])
        fig.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.97,
            text="⚠ Axis capped at 150",
            showarrow=False,
            font=dict(size=13, color="#DECF3F"),
            align="left",
            bgcolor="rgba(14,17,23,0.82)",
            bordercolor="#DECF3F",
            borderwidth=1,
            borderpad=5,
        )

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
        ["1 — Taxonomy of Machine Spirits",
         "2 — Mixed Market Chaos",
         "3 — The Adaptation Duel"],
        label_visibility="collapsed",
    )


# ---------------------------------------------------------------------------
# Page 1: A taxonomy of Machine Spirits
# ---------------------------------------------------------------------------

def page_spirit_gallery(prices: pd.DataFrame, meta: pd.DataFrame) -> None:
    st.title("Taxonomy of Machine Spirits")
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

    # Use a fixed default run whenever the model changes; reset playback.
    if st.session_state.get("p1_group") != chosen_group:
        st.session_state["p1_group"]   = chosen_group
        group_runs = meta[meta.model_group == chosen_group]
        default_id = DEFAULT_RUN_IDS.get(chosen_group)
        if default_id is None:
            default_id = int(random.choice(group_runs.run_id.tolist()))
        st.session_state["p1_run_id"]  = default_id
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
        f"fundamental value $p_f$ = {FUNDAMENTAL_PRICE:.0f}."
    )

    advance_playback("p1", t)


# ---------------------------------------------------------------------------
# Page 2: Mixed Market Chaos
# ---------------------------------------------------------------------------


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
    if "chaos_cat_counts" not in st.session_state:
        st.session_state["chaos_cat_counts"] = {l: 0 for l in CAT_LABELS}

    cat_map = chaos_category_map(prices, meta)

    col_btn, col_caption = st.columns([1, 4])
    with col_btn:
        if st.button("🎲  Run Chaos Roulette", type="primary", use_container_width=True):
            new_run_id = int(random.choice(chaos_runs.run_id.tolist()))
            st.session_state["chaos_run_id"] = new_run_id
            # Reset playback and auto-play so the new run animates from the start
            st.session_state["p2_t"]       = 0
            st.session_state["p2_playing"] = True
            # Increment category count for this run
            cat = cat_map.get(new_run_id, CAT_LABELS[0])
            st.session_state["chaos_cat_counts"][cat] += 1
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
    if t == 49:
        cat = cat_map.get(int(run_id), "")
        if cat in CAT_LABELS:
            cat_color = CAT_COLORS[CAT_LABELS.index(cat)]
            fig.add_annotation(
                xref="paper", yref="paper", x=0.98, y=0.96,
                text=f"<b>{cat.replace(chr(10), '  ')}</b>",
                showarrow=False,
                font=dict(size=22, color=cat_color),
                align="right",
                bgcolor="rgba(14,17,23,0.88)",
                bordercolor=cat_color,
                borderwidth=2,
                borderpad=10,
            )
    st.plotly_chart(fig, width="stretch")

    advance_playback("p2", t)

    st.divider()
    st.subheader("Outcome distribution")
    c_sample, c_reset = st.columns([2, 1])
    with c_sample:
        if st.button("⚡ Sample 500 runs", help="Draw 500 random seeds and tally their outcomes instantly", key="sample_500"):
            run_ids = chaos_runs.run_id.tolist()
            for rid in random.choices(run_ids, k=500):
                cat = cat_map.get(int(rid), CAT_LABELS[0])
                st.session_state["chaos_cat_counts"][cat] += 1
            st.rerun()
    with c_reset:
        if st.button("↺ Reset", help="Reset counts", key="reset_cat_counts"):
            st.session_state["chaos_cat_counts"] = {l: 0 for l in CAT_LABELS}
            st.rerun()
    st.plotly_chart(
        render_category_bar_chart(st.session_state["chaos_cat_counts"]),
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Page 3: The Adaptation Duel
# ---------------------------------------------------------------------------

def page_adaptation_duel(prices: pd.DataFrame, meta: pd.DataFrame) -> None:
    st.title("The Adaptation Duel")
    st.caption(
        "Five Qwen-3 14B trend-followers, plus one adaptive agent. Hit ▶ to watch "
        "three different markets unfold and see the impact of that one adaptive agent."
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
