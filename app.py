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
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
ACTUAL_COLOR      = "#1A1A2E"
FUNDAMENTAL_COLOR = "#16A34A"
AGENT_PALETTE     = ["#5DA5DA", "#FAA43A", "#60BD68", "#F17CB0", "#B276B2", "#DECF3F"]
BG_COLOR          = "#FFFFFF"
GRID_COLOR        = "#E2E8F0"

GEMINI_COLOR      = "#FF5000"
GPT5_COLOR        = "#00B4FF"
QWEN_COLOR        = "#6B7280"

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
BUBBLE_PRICE_THRESHOLD  = 300
BUBBLE_CONSEC_STEPS     = 3
BUBBLE_STD_THRESHOLD    = 100
NO_BUBBLE_STD_THRESHOLD = 20

CAT_LABELS = [
    "No bubble\n(low volatility)",
    "No bubble\n(volatility)",
    "Bubble\n(early volatility)",
    "Bubble\n(late volatility)",
    "Bubble\n(persistent volatility)",
]
CAT_COLORS = ["#60BD68", "#DECF3F", "#FAA43A", "#E05530", "#B03020"]

# Mapping verified empirically from parquet data (t=0 prediction distributions):
#   agent_id=0 always predicts 50.0 (std=0) → OLMo
#   agent_id=5 always predicts 60.0 (std=0) → Gemini (wins 33/50 runs)
#   agent_id=3 predicts 60.0 in 64% of runs, std=9.8 → GPT-5 (erratic)
#   agent_id=2 medium-high std (6.5), mostly 50 → DeepSeek
#   agents 1/4 → Gemma/Qwen by std profile (1=3.1, 4=2.0 ≈ Qwen 2.2)
CHAOS_AGENT_NAMES: dict[int, str] = {
    0: "OLMo-3 7B Think",
    1: "Gemma-3 27B",
    2: "DeepSeek-R1 Llama 8B",
    3: "GPT-5 mini",
    4: "Qwen-3 14B",
    5: "Gemini-3 Flash",
}
CHAOS_AGENT_COLORS: dict[int, str] = {
    0: "#FAA43A",   # OLMo — amber
    1: "#B276B2",   # Gemma — purple
    2: "#60BD68",   # DeepSeek — green
    3: "#00B4FF",   # GPT-5 — ice blue (matches Page 3)
    4: "#5DA5DA",   # Qwen — steel blue
    5: "#FF5000",   # Gemini — fire orange (matches Page 3)
}


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
      h1, h2, h3 { letter-spacing: -0.02em; }
      [data-testid="stMetricValue"] { font-family: "SF Mono", "Menlo", monospace; }
      [data-testid="stMetricLabel"] {
          text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.75rem;
      }
      .ticker-pill {
          display: inline-block; padding: 2px 10px; border-radius: 999px;
          background: #E2E8F0; color: #4A5568; font-family: "SF Mono", monospace;
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
    meta   = meta[meta["filename"].str.match(r"^res_seed_\d+\.pkl$", na=False)]
    prices = prices[prices["run_id"].isin(meta["run_id"])]
    return prices, meta


def display_name(model_group: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_group, model_group)


@st.cache_data(show_spinner=False)
def actual_series(prices: pd.DataFrame, run_id: int) -> pd.DataFrame:
    sub = prices[prices.run_id == run_id]
    anchor = sub.agent_id.min()
    return (sub[sub.agent_id == anchor]
            .sort_values("time_step")[["time_step", "actual_price"]]
            .reset_index(drop=True))


@st.cache_data(show_spinner=False)
def predicted_pivot(prices: pd.DataFrame, run_id: int) -> pd.DataFrame:
    sub = prices[prices.run_id == run_id]
    return (sub.pivot(index="time_step", columns="agent_id", values="predicted_price")
            .sort_index())


def has_bubble(price_series: "pd.Series") -> bool:
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


@st.cache_data(show_spinner=False)
def compute_earnings_df(prices: pd.DataFrame, run_id: int) -> pd.DataFrame:
    """Per-agent, per-step earnings using the quadratic scoring rule from the paper.

    e_{h,t} = max( 1300 - (1300/49) * (p_t - p^e_{h,t})^2, 0 )
    """
    sub = prices[prices.run_id == run_id][
        ["agent_id", "time_step", "predicted_price", "actual_price"]
    ].copy()
    sub["earnings"] = (
        1300 - (1300 / 49) * (sub["actual_price"] - sub["predicted_price"]) ** 2
    ).clip(lower=0)
    return sub[["agent_id", "time_step", "earnings"]]


def render_category_bar_chart(counts: dict[str, int]) -> go.Figure:
    total = sum(counts.values())
    order  = sorted(range(len(CAT_LABELS)), key=lambda i: counts.get(CAT_LABELS[i], 0), reverse=True)
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
        title_font=dict(size=13, color="#1A1A2E"),
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_layout(layout)
    return fig


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def base_layout(height: int = 480) -> dict:
    return dict(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color="#1A1A2E", family="Inter, system-ui, sans-serif"),
        margin=dict(l=50, r=20, t=20, b=90),
        height=height,
        xaxis=dict(
            title="Time step", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
            range=[0, 49],
        ),
        yaxis=dict(title="Price", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=GRID_COLOR, borderwidth=1,
        ),
        hovermode="x unified",
    )


def add_fundamental_line(fig: go.Figure, row=None, col=None) -> None:
    kw: dict = dict(
        y=FUNDAMENTAL_PRICE,
        line_dash="dash",
        line_color=FUNDAMENTAL_COLOR,
        opacity=0.6,
        annotation_text=f"  p_f = {FUNDAMENTAL_PRICE:.0f}",
        annotation_position="top right",
        annotation=dict(font=dict(color=FUNDAMENTAL_COLOR, size=11)),
    )
    if row is not None:
        kw["row"] = row
    if col is not None:
        kw["col"] = col
    fig.add_hline(**kw)


def _animation_controls(fig: go.Figure) -> None:
    """Add Plotly-native play/pause buttons and a period slider to fig."""
    slider_steps = [
        dict(
            args=[[str(t)], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
            label=str(t) if (t % 10 == 0 or t == 49) else "",
            method="animate",
        )
        for t in range(50)
    ]
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            direction="left",
            pad=dict(r=10, t=87),
            x=0.1, xanchor="right",
            y=0.0, yanchor="top",
            buttons=[
                dict(
                    label="▶  Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=150, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=0),
                    )],
                ),
                dict(
                    label="⏸  Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0),
                    )],
                ),
            ],
        )],
        sliders=[dict(
            active=0,
            steps=slider_steps,
            currentvalue=dict(
                prefix="Period: ",
                visible=True,
                xanchor="center",
                font=dict(size=13, color="#1A1A2E"),
            ),
            transition=dict(duration=0),
            pad=dict(b=10, t=50),
            len=0.9,
            x=0.1, xanchor="left",
            y=0.0, yanchor="top",
            bgcolor=GRID_COLOR,
            bordercolor="#CBD5E0",
            tickcolor="#CBD5E0",
            font=dict(color="#1A1A2E", size=10),
        )],
    )


def build_animated_figure(
    prices: pd.DataFrame,
    run_id: int,
    show_predictions: bool,
    dynamic_yaxis: bool = False,
    earnings_df: pd.DataFrame | None = None,
    cat_label: str | None = None,
    height: int = 480,
) -> go.Figure:
    """Pre-compute all 50 animation frames for smooth client-side playback.

    With earnings_df provided, adds a cumulative earnings subplot (Page 2 only).
    """
    actual    = actual_series(prices, run_id)
    pred      = predicted_pivot(prices, run_id) if show_predictions else None
    agent_ids = list(pred.columns) if pred is not None else []
    n_pred    = len(agent_ids)
    has_earn  = earnings_df is not None

    # ── Figure creation ────────────────────────────────────────────────────────
    if has_earn:
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.65, 0.35],
            vertical_spacing=0.08,
        )
        layout = base_layout(height=height)
        layout.pop("xaxis", None)
        layout.pop("yaxis", None)
        layout["margin"] = dict(l=50, r=20, t=70, b=170)  # t=70 leaves room above legend for category label
        fig.update_layout(**layout)
        fig.update_xaxes(title="Time step", gridcolor=GRID_COLOR,
                         zerolinecolor=GRID_COLOR, range=[0, 49], row=1, col=1)
        fig.update_yaxes(title="Price ($)", gridcolor=GRID_COLOR,
                         zerolinecolor=GRID_COLOR, row=1, col=1)
        fig.update_xaxes(title="Cumulative earnings (pts)", gridcolor=GRID_COLOR,
                         zerolinecolor=GRID_COLOR, range=[0, 65_000], row=2, col=1)
        fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, row=2, col=1)
    else:
        layout = base_layout(height=height)
        layout["margin"] = dict(l=50, r=20, t=20, b=170)
        fig = go.Figure(layout=layout)

    # ── Initial traces (t=0 state) ─────────────────────────────────────────────
    t0_actual = actual[actual.time_step <= 0]
    t0_pred   = pred[pred.index <= 0] if pred is not None else None

    for i, agent_id in enumerate(agent_ids):
        tr = go.Scatter(
            x=t0_pred.index.tolist() if t0_pred is not None else [],
            y=t0_pred[agent_id].tolist() if t0_pred is not None else [],
            mode="lines",
            name=f"Agent {agent_id} forecast",
            line=dict(color=AGENT_PALETTE[i % len(AGENT_PALETTE)], width=1),
            opacity=0.55,
        )
        if has_earn:
            fig.add_trace(tr, row=1, col=1)
        else:
            fig.add_trace(tr)

    price_tr = go.Scatter(
        x=t0_actual.time_step.tolist(),
        y=t0_actual.actual_price.tolist(),
        mode="lines",
        name="Actual price",
        line=dict(color=ACTUAL_COLOR, width=3),
    )
    if has_earn:
        fig.add_trace(price_tr, row=1, col=1)
    else:
        fig.add_trace(price_tr)

    if has_earn:
        for agent_id in range(6):
            fig.add_trace(go.Bar(
                x=[0.0],
                y=[CHAOS_AGENT_NAMES[agent_id]],
                orientation="h",
                name=CHAOS_AGENT_NAMES[agent_id],
                marker_color=CHAOS_AGENT_COLORS[agent_id],
                showlegend=False,
                textposition="outside",
                cliponaxis=False,
            ), row=2, col=1)

    # ── Static elements ────────────────────────────────────────────────────────
    if has_earn:
        add_fundamental_line(fig, row=1, col=1)
    else:
        add_fundamental_line(fig)

    if not dynamic_yaxis:
        run_max = float(actual.actual_price.max())
        y_range = [0, 1000] if run_max > 150 else [0, 150]
        if has_earn:
            fig.update_yaxes(range=y_range, row=1, col=1)
        else:
            fig.update_layout(yaxis_range=y_range)
        if run_max <= 150:
            fig.add_annotation(
                xref="paper", yref="paper", x=0.01, y=0.97,
                text="⚠ Axis capped at 150",
                showarrow=False,
                font=dict(size=13, color="#DECF3F"),
                align="left",
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="#DECF3F", borderwidth=1, borderpad=5,
            )

    # ── Build animation frames ─────────────────────────────────────────────────
    n_traces   = n_pred + 1 + (6 if has_earn else 0)
    all_traces = list(range(n_traces))

    frames = []
    for t in range(50):
        actual_t   = actual[actual.time_step <= t]
        frame_data = []

        if pred is not None:
            pred_t = pred[pred.index <= t]
            for agent_id in agent_ids:
                frame_data.append(go.Scatter(
                    x=pred_t.index.tolist(),
                    y=pred_t[agent_id].tolist(),
                ))

        current_price = float(actual_t.actual_price.iloc[-1]) if not actual_t.empty else 0.0
        if dynamic_yaxis:
            if current_price > BUBBLE_PRICE_THRESHOLD:
                lc, lw = "#FF5000", 4
            elif current_price > 200:
                lc, lw = "#FAA43A", 3.5
            else:
                lc, lw = ACTUAL_COLOR, 3
        else:
            lc, lw = ACTUAL_COLOR, 3

        frame_data.append(go.Scatter(
            x=actual_t.time_step.tolist(),
            y=actual_t.actual_price.tolist(),
            line=dict(color=lc, width=lw),
        ))

        if has_earn:
            cumul = (earnings_df[earnings_df.time_step <= t]
                     .groupby("agent_id")["earnings"].sum())
            for agent_id in range(6):
                val = float(cumul.get(agent_id, 0.0))
                frame_data.append(go.Bar(
                    x=[val],
                    y=[CHAOS_AGENT_NAMES[agent_id]],
                    text=[f"{val:,.0f}"],
                ))

        frame_annotations: list[dict] = []
        frame_layout: dict = {"annotations": frame_annotations}

        if dynamic_yaxis:
            t_max = float(actual_t.actual_price.max()) if not actual_t.empty else 0.0
            y_max = max(150.0, t_max * 1.15)
            frame_layout["yaxis"] = {"range": [0, y_max]}

        if t == 49 and cat_label and cat_label in CAT_LABELS:
            cat_color = CAT_COLORS[CAT_LABELS.index(cat_label)]
            frame_annotations.append(dict(
                xref="paper", yref="paper",
                x=0.98, y=1.07, yanchor="bottom",
                text=f"<b>{cat_label.replace(chr(10), '  ')}</b>",
                showarrow=False,
                font=dict(size=18, color=cat_color),
                align="right",
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor=cat_color, borderwidth=2, borderpad=8,
            ))
        elif dynamic_yaxis:
            t_max_val = float(actual_t.actual_price.max()) if not actual_t.empty else 0.0
            if 200 < t_max_val <= BUBBLE_PRICE_THRESHOLD:
                frame_annotations.append(dict(
                    xref="paper", yref="paper", x=0.98, y=0.96,
                    text="⚡ Price surging...",
                    showarrow=False,
                    font=dict(size=15, color="#FAA43A"),
                    align="right",
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#FAA43A", borderwidth=1, borderpad=6,
                ))

        frames.append(go.Frame(
            data=frame_data,
            layout=frame_layout,
            name=str(t),
            traces=all_traces,
        ))

    fig.frames = frames
    _animation_controls(fig)
    return fig


def build_animated_duel_figure(prices: pd.DataFrame, height: int = 560) -> go.Figure:
    """Animated overlay of the three duel runs — all frames pre-computed client-side."""
    qwen = actual_series(prices, QWEN_BASELINE_RUN_ID)
    gem  = actual_series(prices, GEMINI_HERO_RUN_ID)
    gpt  = actual_series(prices, GPT5_HERO_RUN_ID)

    layout = base_layout(height=height)
    layout["margin"] = dict(l=50, r=20, t=20, b=170)
    fig = go.Figure(layout=layout)

    t0_q = qwen[qwen.time_step <= 0]
    t0_g = gem[gem.time_step <= 0]
    t0_p = gpt[gpt.time_step <= 0]

    # Trace 0: Qwen baseline
    fig.add_trace(go.Scatter(
        x=t0_q.time_step.tolist(), y=t0_q.actual_price.tolist(),
        mode="lines",
        name="🧭  6x Qwen-3 14B — Baseline",
        line=dict(color=QWEN_COLOR, width=2.5, dash="dot"),
    ))
    # Trace 1: Gemini glow shadow
    fig.add_trace(go.Scatter(
        x=t0_g.time_step.tolist(), y=t0_g.actual_price.tolist(),
        mode="lines", showlegend=False, hoverinfo="skip",
        line=dict(color="rgba(255,80,0,0.15)", width=16),
    ))
    # Trace 2: Gemini actual
    fig.add_trace(go.Scatter(
        x=t0_g.time_step.tolist(), y=t0_g.actual_price.tolist(),
        mode="lines",
        name="🔥  1x Gemini-3 Flash vs 5x Qwen-3 14B",
        line=dict(color=GEMINI_COLOR, width=3),
    ))
    # Trace 3: GPT glow shadow
    fig.add_trace(go.Scatter(
        x=t0_p.time_step.tolist(), y=t0_p.actual_price.tolist(),
        mode="lines", showlegend=False, hoverinfo="skip",
        line=dict(color="rgba(0,180,255,0.15)", width=16),
    ))
    # Trace 4: GPT-5 actual
    fig.add_trace(go.Scatter(
        x=t0_p.time_step.tolist(), y=t0_p.actual_price.tolist(),
        mode="lines",
        name="🧊  1x GPT-5 mini vs 5x Qwen-3 14B",
        line=dict(color=GPT5_COLOR, width=3),
    ))

    run_max = max(
        float(qwen.actual_price.max()),
        float(gem.actual_price.max()),
        float(gpt.actual_price.max()),
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
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#DECF3F", borderwidth=1, borderpad=5,
        )

    add_fundamental_line(fig)

    frames = []
    for t in range(50):
        qt = qwen[qwen.time_step <= t]
        gt = gem[gem.time_step <= t]
        pt = gpt[gpt.time_step <= t]
        frames.append(go.Frame(
            data=[
                go.Scatter(x=qt.time_step.tolist(), y=qt.actual_price.tolist()),
                go.Scatter(x=gt.time_step.tolist(), y=gt.actual_price.tolist()),
                go.Scatter(x=gt.time_step.tolist(), y=gt.actual_price.tolist()),
                go.Scatter(x=pt.time_step.tolist(), y=pt.actual_price.tolist()),
                go.Scatter(x=pt.time_step.tolist(), y=pt.actual_price.tolist()),
            ],
            name=str(t),
            traces=[0, 1, 2, 3, 4],
        ))

    fig.frames = frames
    _animation_controls(fig)
    return fig


# ---------------------------------------------------------------------------
# Page 0: Introduction
# ---------------------------------------------------------------------------

def page_intro(prices: pd.DataFrame, meta: pd.DataFrame) -> None:  # noqa: ARG001
    st.title("Machine Spirits")
    st.subheader("A Simulated Market of LLM Agents")

    st.markdown(
        """
        This dashboard is an interactive companion to the paper
        **[Machine Spirits: Speculation and Adaptation of LLM Agents in Asset Markets](https://arxiv.org/abs/2604.18602)**.

        ---

        ### How does the market work?

        Six AI language models (LLMs) each act as a trader in a simple financial market.
        Every period (round), each agent submits its **forecast** of what the asset price
        will be next period.  The actual market price that emerges is then the average of
        all six forecasts, plus a fixed dividend — so the price is literally *made of
        beliefs*.

        Because today's price depends on what agents expect tomorrow, and tomorrow's
        expectations react to today's price, the market has a built-in **feedback loop**.
        This loop is what allows prices to spiral far above the true fundamental value —
        a *bubble* — or converge smoothly towards it, depending on how each model reasons.

        ### What is the fundamental value?

        The asset has a true underlying value of **$60** (marked as the dashed green line
        in all charts).  A perfectly rational agent would always forecast $60 and the
        market would never deviate.  In practice, LLMs behave very differently from one
        another — some anchor close to $60, others trend-chase and amplify deviations.

        ### What can I explore?
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(
            "**📊 Taxonomy of Machine Spirits**\n\n"
            "Pick a model and watch 6 copies of it trade together. "
            "How does each LLM's economic 'personality' shape the market?"
        )
    with col2:
        st.info(
            "**🎲 Mixed Market Chaos**\n\n"
            "Six *different* LLMs share one market. Roll the dice — same models, "
            "different random seeds — and see how wildly outcomes can vary."
        )
    with col3:
        st.info(
            "**⚔️ The Adaptation Duel**\n\n"
            "Five Qwen trend-followers plus one adaptive agent. "
            "Watch how a single model can shift the entire market."
        )

    st.markdown("---")
    st.caption(
        "Use the sidebar on the left to navigate between the three pages. "
        "Each page has playback controls — press ▶ to animate the market evolving step by step."
    )


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

def render_sidebar() -> str:
    st.sidebar.title("Machine Spirits")
    st.sidebar.caption("Speculation & adaptation of LLM agents in asset markets")
    return st.sidebar.radio(
        "Pages",
        ["0 — Introduction",
         "1 — Taxonomy of Machine Spirits",
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

    if chosen_group is None:
        return

    if st.session_state.get("p1_group") != chosen_group:
        st.session_state["p1_group"] = chosen_group
        group_runs = meta[meta.model_group == chosen_group]
        default_id = DEFAULT_RUN_IDS.get(chosen_group)
        if default_id is None:
            default_id = int(random.choice(group_runs.run_id.tolist()))
        st.session_state["p1_run_id"] = default_id

    chosen_run = st.session_state["p1_run_id"]

    show_pred = st.checkbox(
        "Show agent forecasts", value=True,
        help="Overlay each of the 6 agents' next-period price expectations.",
    )

    fig = build_animated_figure(prices, chosen_run, show_predictions=show_pred)
    st.plotly_chart(fig, width='stretch')

    st.caption(
        "Bold line: the realized market price. Thin lines: each agent's "
        f"submitted forecast for the next period. The dashed line marks the "
        f"fundamental value $p_f$ = {FUNDAMENTAL_PRICE:.0f}."
    )


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
        if st.button("🎲  Run Chaos Roulette", type="primary", width='stretch'):
            new_run_id = int(random.choice(chaos_runs.run_id.tolist()))
            st.session_state["chaos_run_id"] = new_run_id
            st.session_state["chaos_autoplay"] = True
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
        f"<div style='margin: 4px 0 -8px 4px; color:#4A5568;'>"
        f"<span class='ticker-pill'>Seed {int(row.seed)}</span></div>",
        unsafe_allow_html=True,
    )

    earnings_df = compute_earnings_df(prices, int(run_id))
    cat_label   = cat_map.get(int(run_id))

    fig = build_animated_figure(
        prices, int(run_id),
        show_predictions=show_pred,
        dynamic_yaxis=True,
        earnings_df=earnings_df,
        cat_label=cat_label,
        height=600,
    )
    st.plotly_chart(fig, width='stretch')

    # Auto-play when a new run is selected via Roulette
    if st.session_state.get("chaos_autoplay", False):
        st.session_state["chaos_autoplay"] = False
        import streamlit.components.v1 as components
        components.html("""
        <script>
        (function() {
            var tries = 0;
            function tryPlay() {
                tries++;
                var plots = window.parent.document.querySelectorAll('.js-plotly-plot');
                var found = false;
                plots.forEach(function(plot) {
                    if (plot._frames && plot._frames.length > 1 && window.parent.Plotly) {
                        window.parent.Plotly.animate(plot, null, {
                            frame: {duration: 150, redraw: true},
                            fromcurrent: false,
                            transition: {duration: 0}
                        });
                        found = true;
                    }
                });
                if (!found && tries < 15) { setTimeout(tryPlay, 200); }
            }
            setTimeout(tryPlay, 400);
        })();
        </script>
        """, height=0)

    # Winner callout — always visible (not gated on t=49)
    cumul     = earnings_df.groupby("agent_id")["earnings"].sum()
    winner_id = int(cumul.idxmax())
    st.success(
        f"🏆 Winner (all 50 periods): **{CHAOS_AGENT_NAMES.get(winner_id, f'Agent {winner_id}')}** "
        f"— {int(cumul.max()):,} pts"
    )

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
        width='stretch',
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

    fig = build_animated_duel_figure(prices)
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    prices, meta = load_data()
    page = render_sidebar()
    if page.startswith("0"):
        page_intro(prices, meta)
    elif page.startswith("1"):
        page_spirit_gallery(prices, meta)
    elif page.startswith("2"):
        page_chaos(prices, meta)
    else:
        page_adaptation_duel(prices, meta)


if __name__ == "__main__":
    main()
