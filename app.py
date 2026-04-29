"""Machine Spirits — Interactive Dashboard.

Companion to "Machine Spirits: Speculation and Adaptation of LLM Agents in
Asset Markets". Three pages:

  1. The Spirit Gallery   — homogeneous single-LLM markets.
  2. Mixed Market Chaos   — seed-driven divergence in a 6-model mixed market.
  3. The Adaptation Duel  — Gemini-3-Flash vs GPT-5-mini, both placed with
                            5x Qwen-3-14B trend-followers, anchored against
                            the pure 6x Qwen baseline.

Run:   streamlit run app.py
Data:  dashboard_data.parquet + runs_meta.parquet (produced by prepare_data.py).
"""

from __future__ import annotations

import random
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
VOLATILITY_THRESHOLD = 100.0  # std cutoff for "volatile" badge

CHAOS_GROUP = (
    "1Qwen-Qwen3-14B_and_1allenai-Olmo-3-7B-Think"
    "_and_1deepseek-ai-DeepSeek-R1-Distill-Llama-8B"
    "_and_1gemini-3-flash-preview_and_1gemma-3-27b-it_and_1gpt-5-mini"
)
QWEN_GROUP = "6Qwen-Qwen3-14B"
GEMINI_GROUP = "5Qwen-Qwen3-14B_and_1gemini-3-flash-preview"
GPT5_GROUP = "5Qwen-Qwen3-14B_and_1gpt-5-mini"

# Hand-picked hero runs (chosen from runs_meta.parquet stats).
QWEN_BASELINE_RUN_ID = 46    # typical Qwen: peak ~954 then settles ~75
GEMINI_HERO_RUN_ID = 295     # two distinct bubbles; IQR=828, peak=948
GPT5_HERO_RUN_ID = 7         # one bubble then settles near fundamental; late_std=13

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "6Qwen-Qwen3-14B": "Qwen-3 14B (x6)",
    "6Qwen-Qwen3-14B_no_reasoning": "Qwen-3 14B, no reasoning (x6)",
    "6Qwen-Qwen3-32B": "Qwen-3 32B (x6)",
    "6Qwen-Qwen2-5-7B-Instruct": "Qwen-2.5 7B Instruct (x6)",
    "6gemini-2-5-flash": "Gemini-2.5 Flash (x6)",
    "6gemini-3-flash-preview": "Gemini-3 Flash preview (x6)",
    "6gemma-3-27b-it": "Gemma-3 27B Instruct (x6)",
    "6gpt-4-1": "GPT-4.1 (x6)",
    "6gpt-4o-mini": "GPT-4o mini (x6)",
    "6gpt-5-mini": "GPT-5 mini (x6)",
    "6o3": "OpenAI o3 (x6)",
    "6o3-mini": "OpenAI o3-mini (x6)",
    "6allenai-Olmo-3-7B-Instruct": "OLMo-3 7B Instruct (x6)",
    "6allenai-Olmo-3-7B-Think": "OLMo-3 7B Think (x6)",
    "6deepseek-ai-DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1 Distill Llama 8B (x6)",
    "6deepseek-ai-DeepSeek-R1-Distill-Qwen-14B": "DeepSeek-R1 Distill Qwen 14B (x6)",
    QWEN_GROUP: "Qwen-3 14B (x6)",
    GEMINI_GROUP: "5x Qwen-3 14B + 1x Gemini-3 Flash",
    GPT5_GROUP: "5x Qwen-3 14B + 1x GPT-5 mini",
    CHAOS_GROUP: "Mixed (Qwen + OLMo + DeepSeek + Gemini + Gemma + GPT-5)",
}

# Plotly fintech palette
ACTUAL_COLOR = "#F2F2F2"
FUNDAMENTAL_COLOR = "#7CFFA1"
AGENT_PALETTE = [
    "#5DA5DA", "#FAA43A", "#60BD68", "#F17CB0",
    "#B276B2", "#DECF3F",
]
BG_COLOR = "#0E1117"
GRID_COLOR = "#1C2230"


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
      [data-testid="stMetricLabel"] { text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.75rem; }
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
    prices = pd.read_parquet(DATA_FILE)
    meta = pd.read_parquet(META_FILE)
    return prices, meta


def display_name(model_group: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_group, model_group)


def compute_iqr(series: pd.Series) -> float:
    return float(series.quantile(0.75) - series.quantile(0.25))


def actual_series(prices: pd.DataFrame, run_id: int) -> pd.DataFrame:
    sub = prices[prices.run_id == run_id]
    anchor = sub.agent_id.min()
    return (sub[sub.agent_id == anchor]
            .sort_values("time_step")[["time_step", "actual_price"]]
            .reset_index(drop=True))


def predicted_pivot(prices: pd.DataFrame, run_id: int) -> pd.DataFrame:
    """Wide format: index=time_step, columns=agent_id, values=predicted_price."""
    sub = prices[prices.run_id == run_id]
    return (sub.pivot(index="time_step", columns="agent_id",
                      values="predicted_price")
            .sort_index())


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def base_layout(title: str, height: int = 460) -> dict:
    return dict(
        title=dict(text=title, x=0.0, font=dict(size=18, color="#E5E9F0")),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color="#C8CDD7", family="Inter, system-ui, sans-serif"),
        margin=dict(l=40, r=20, t=60, b=40),
        height=height,
        xaxis=dict(
            title="Time step", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
            range=[0, 49],
        ),
        yaxis=dict(
            title="Price", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0,
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


def static_run_figure(
    prices: pd.DataFrame,
    run_id: int,
    title: str,
    show_predictions: bool,
    height: int = 460,
) -> go.Figure:
    """Static (non-animated) chart: actual + 6 agent prediction lines."""
    actual = actual_series(prices, run_id)
    fig = go.Figure(layout=base_layout(title, height=height))

    if show_predictions:
        pred = predicted_pivot(prices, run_id)
        for i, agent_id in enumerate(pred.columns):
            fig.add_trace(go.Scatter(
                x=pred.index, y=pred[agent_id],
                mode="lines",
                name=f"Agent {agent_id} forecast",
                line=dict(color=AGENT_PALETTE[i % len(AGENT_PALETTE)], width=1),
                opacity=0.55,
            ))

    fig.add_trace(go.Scatter(
        x=actual.time_step, y=actual.actual_price,
        mode="lines",
        name="Actual price",
        line=dict(color=ACTUAL_COLOR, width=3),
    ))

    add_fundamental_line(fig)
    return fig


def animated_run_figure(
    prices: pd.DataFrame,
    run_id: int,
    title: str,
    show_predictions: bool,
    height: int = 520,
) -> go.Figure:
    """Animated chart that progressively reveals actual + agent forecasts."""
    actual = actual_series(prices, run_id)
    pred = predicted_pivot(prices, run_id) if show_predictions else None

    n_steps = int(actual.time_step.max()) + 1
    y_max = float(actual.actual_price.max())
    if pred is not None:
        y_max = max(y_max, float(pred.max().max()))
    y_pad = y_max * 0.06 + 5

    initial_actual = actual.iloc[: 1]
    initial_traces = [
        go.Scatter(
            x=initial_actual.time_step, y=initial_actual.actual_price,
            mode="lines", name="Actual price",
            line=dict(color=ACTUAL_COLOR, width=3),
        )
    ]
    if pred is not None:
        for i, agent_id in enumerate(pred.columns):
            initial_traces.append(go.Scatter(
                x=pred.index[:1], y=pred[agent_id].values[:1],
                mode="lines", name=f"Agent {agent_id} forecast",
                line=dict(color=AGENT_PALETTE[i % len(AGENT_PALETTE)], width=1),
                opacity=0.55,
            ))

    frames = []
    for t in range(1, n_steps + 1):
        frame_data = [
            go.Scatter(
                x=actual.time_step.values[:t],
                y=actual.actual_price.values[:t],
                mode="lines", line=dict(color=ACTUAL_COLOR, width=3),
            )
        ]
        if pred is not None:
            for i, agent_id in enumerate(pred.columns):
                frame_data.append(go.Scatter(
                    x=pred.index.values[:t], y=pred[agent_id].values[:t],
                    mode="lines",
                    line=dict(color=AGENT_PALETTE[i % len(AGENT_PALETTE)], width=1),
                    opacity=0.55,
                ))
        frames.append(go.Frame(data=frame_data, name=str(t)))

    layout = base_layout(title, height=height)
    layout["yaxis"]["range"] = [-5, y_max + y_pad]
    layout["updatemenus"] = [dict(
        type="buttons",
        showactive=False,
        x=0.01, y=1.18, xanchor="left", yanchor="top",
        bgcolor="#1C2230",
        bordercolor="#1C2230",
        font=dict(color="#E5E9F0"),
        buttons=[
            dict(label="▶  Play",
                 method="animate",
                 args=[None, dict(frame=dict(duration=80, redraw=True),
                                  fromcurrent=True,
                                  transition=dict(duration=0))]),
            dict(label="❚❚  Pause",
                 method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                    transition=dict(duration=0))]),
        ],
    )]
    layout["sliders"] = [dict(
        active=0, x=0.12, y=1.10, len=0.85,
        currentvalue=dict(prefix="t = ", font=dict(color="#C8CDD7")),
        bgcolor="#1C2230",
        steps=[dict(method="animate",
                    args=[[str(t)], dict(mode="immediate",
                                          frame=dict(duration=0, redraw=True),
                                          transition=dict(duration=0))],
                    label=str(t))
               for t in range(1, n_steps + 1)],
    )]

    fig = go.Figure(data=initial_traces, layout=layout, frames=frames)
    add_fundamental_line(fig)
    return fig


# ---------------------------------------------------------------------------
# Sidebar: research context + page nav
# ---------------------------------------------------------------------------

def render_sidebar() -> str:
    st.sidebar.title("Machine Spirits")
    st.sidebar.caption("Speculation & adaptation of LLM agents in asset markets")

    page = st.sidebar.radio(
        "Pages",
        ["1 — The Spirit Gallery",
         "2 — Mixed Market Chaos",
         "3 — The Adaptation Duel"],
        label_visibility="collapsed",
    )

    with st.sidebar.expander("Research context", expanded=False):
        st.markdown(
            "Six LLM agents trade a single asset over **50 periods**. Each "
            "period, every agent submits an expected next-period price. The "
            "market clears at:"
        )
        st.latex(
            r"p_t \;=\; \frac{1}{1+r}\left(\frac{1}{6}\sum_{h=1}^{6} "
            r"p_{h,t+1}^{e} \;+\; \bar{y}\right)"
        )
        st.markdown(
            f"The **fundamental price** is **p_f = {FUNDAMENTAL_PRICE:.0f}**. "
            "Because tomorrow's price depends on the average of today's "
            "expectations, optimism is self-fulfilling — small upward biases "
            "compound into bubbles. Different LLMs anchor very differently, "
            "and a single non-trend-following agent in a mixed market can "
            "shift the entire dynamic."
        )

    st.sidebar.markdown("---")
    st.sidebar.caption("Press R to rerun · ⓘ to view source")
    return page


# ---------------------------------------------------------------------------
# Page 1: The Spirit Gallery
# ---------------------------------------------------------------------------

def page_spirit_gallery(prices: pd.DataFrame, meta: pd.DataFrame) -> None:
    st.title("The Spirit Gallery")
    st.caption(
        "Each LLM has its own economic disposition. Pick a homogeneous market "
        "(six copies of the same model) and watch how the price evolves."
    )

    homogeneous = sorted(
        meta[~meta.model_group.str.contains("_and_") &
             meta.model_group.str.startswith("6")]
        .model_group.unique()
    )

    col_model, col_seed = st.columns([3, 2])
    with col_model:
        chosen_group = st.selectbox(
            "Market composition",
            options=homogeneous,
            format_func=display_name,
            index=homogeneous.index(QWEN_GROUP) if QWEN_GROUP in homogeneous else 0,
        )

    group_runs = meta[meta.model_group == chosen_group].sort_values("seed")
    median_iqr = group_runs.iqr.median()
    default_run = (group_runs.assign(d=(group_runs.iqr - median_iqr).abs())
                              .sort_values("d").iloc[0].run_id)

    with col_seed:
        chosen_run = st.selectbox(
            "Seed / run",
            options=group_runs.run_id.tolist(),
            format_func=lambda rid: (
                f"seed {int(group_runs.loc[group_runs.run_id == rid, 'seed'].iloc[0]):>2d}"
                f"   (peak {group_runs.loc[group_runs.run_id == rid, 'peak_price'].iloc[0]:.0f},"
                f"  IQR {group_runs.loc[group_runs.run_id == rid, 'iqr'].iloc[0]:.0f})"
            ),
            index=int(group_runs.run_id.tolist().index(int(default_run))),
        )

    show_pred = st.checkbox("Show agent forecasts", value=True,
                            help="Overlay each of the 6 agents' next-period price expectations.")

    fig = animated_run_figure(
        prices, chosen_run,
        title=f"{display_name(chosen_group)} — actual price over 50 periods",
        show_predictions=show_pred,
    )
    st.plotly_chart(fig, width="stretch")

    row = group_runs[group_runs.run_id == chosen_run].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak price", f"{row.peak_price:.1f}")
    m2.metric("Mean price", f"{row.mean_price:.1f}")
    m3.metric("IQR (price)", f"{row.iqr:.1f}")
    m4.metric("Late std", f"{row.late_std:.1f}")

    st.caption(
        "Bold line: the realized market price. Thin lines: each agent's "
        f"submitted forecast for the next period. The dashed line marks the "
        f"fundamental value p_f = {FUNDAMENTAL_PRICE:.0f}. Hit ▶ to animate."
    )


# ---------------------------------------------------------------------------
# Page 2: Mixed Market Chaos
# ---------------------------------------------------------------------------

def volatility_label(early_std: float, late_std: float) -> tuple[str, str]:
    """Return (label, severity) where severity is 'success' | 'warning' | 'error'."""
    early_hi = early_std > VOLATILITY_THRESHOLD
    late_hi = late_std > VOLATILITY_THRESHOLD
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

    state_key = "chaos_run_id"
    if state_key not in st.session_state:
        st.session_state[state_key] = int(chaos_runs.iloc[0].run_id)

    col_btn, col_caption = st.columns([1, 4])
    with col_btn:
        if st.button("🎲  Run Chaos Roulette", type="primary", width="stretch"):
            st.session_state[state_key] = int(random.choice(chaos_runs.run_id.tolist()))
    with col_caption:
        st.markdown(
            f"<span class='ticker-pill'>{len(chaos_runs)} seeds available</span>",
            unsafe_allow_html=True,
        )

    run_id = st.session_state[state_key]
    row = chaos_runs[chaos_runs.run_id == run_id].iloc[0]

    show_pred = st.checkbox("Show agent forecasts", value=True, key="chaos_show_pred")
    fig = static_run_figure(
        prices, int(run_id),
        title=f"Seed {int(row.seed)}  ·  {row.filename}",
        show_predictions=show_pred,
        height=480,
    )
    st.plotly_chart(fig, width="stretch")

    label, severity = volatility_label(row.early_std, row.late_std)
    m1, m2, m3 = st.columns(3)
    m1.metric("Early std (t < 25)", f"{row.early_std:.1f}",
              delta=f"{'volatile' if row.early_std > VOLATILITY_THRESHOLD else 'calm'}",
              delta_color="off")
    m2.metric("Late std (t ≥ 25)", f"{row.late_std:.1f}",
              delta=f"{'volatile' if row.late_std > VOLATILITY_THRESHOLD else 'calm'}",
              delta_color="off")
    m3.metric("Peak price", f"{row.peak_price:.1f}")

    badge_fn = {"success": st.success, "warning": st.warning, "error": st.error}[severity]
    badge_fn(f"**Regime:** {label}  ·  threshold = std > {VOLATILITY_THRESHOLD:.0f}")


# ---------------------------------------------------------------------------
# Page 3: The Adaptation Duel
# ---------------------------------------------------------------------------

def page_adaptation_duel(prices: pd.DataFrame, meta: pd.DataFrame) -> None:
    st.title("The Adaptation Duel")
    st.caption(
        "Five Qwen-3 14B trend-followers, plus one adaptive agent. Swap that "
        "single agent and the market changes character. The center column "
        "shows what pure Qwen does on its own — the substrate the next two "
        "columns disrupt."
    )

    col_qwen, col_gemini, col_gpt5 = st.columns(3)

    def render_column(col, run_id: int, group: str, title: str, callout_kind: str,
                      callout_title: str, callout_body: str) -> None:
        with col:
            st.subheader(title)
            row = meta[meta.run_id == run_id].iloc[0]
            fig = static_run_figure(
                prices, run_id,
                title=f"Seed {int(row.seed)}  ·  IQR {row.iqr:.0f}",
                show_predictions=True,
                height=380,
            )
            fig.update_layout(showlegend=False, margin=dict(l=30, r=10, t=50, b=30))
            st.plotly_chart(fig, width="stretch")
            st.metric("IQR (this run)", f"{row.iqr:.1f}")
            {"info": st.info, "warning": st.warning, "success": st.success}[callout_kind](
                f"**{callout_title}**\n\n{callout_body}"
            )

    render_column(
        col_qwen, QWEN_BASELINE_RUN_ID, QWEN_GROUP,
        title="Baseline · 6× Qwen-3 14B",
        callout_kind="info",
        callout_title="Pure trend-following",
        callout_body=(
            "Six identical Qwen agents agree on momentum. The market bubbles "
            "high, crashes, and eventually settles back near the fundamental. "
            "This is the substrate the next two columns disrupt."
        ),
    )

    render_column(
        col_gemini, GEMINI_HERO_RUN_ID, GEMINI_GROUP,
        title="5× Qwen + 1× Gemini-3 Flash",
        callout_kind="warning",
        callout_title="🦅  Apex Predator",
        callout_body=(
            "Gemini-3 Flash uses theory-of-mind to model what the trend-followers "
            "will do — and front-runs them. Each time the market starts to "
            "mean-revert, Gemini reignites the bubble. Result: **two distinct "
            "peaks** and the largest IQR of any configuration."
        ),
    )

    render_column(
        col_gpt5, GPT5_HERO_RUN_ID, GPT5_GROUP,
        title="5× Qwen + 1× GPT-5 mini",
        callout_kind="success",
        callout_title="🛡  Volatility damper",
        callout_body=(
            "GPT-5 mini anchors close to the fundamental and refuses to chase "
            "the rally. After one initial bubble episode, its persistent counter-"
            "weight pulls the market back to a tight band near p_f."
        ),
    )

    st.markdown("---")
    st.subheader("Population-level volatility (mean IQR across all seeds)")
    st.caption(
        "The single runs above are extreme examples chosen for visual contrast. "
        "Averaged across every seed in each subset, the volatility ordering is "
        "the same — Gemini tops, Qwen middle, GPT-5 lowest."
    )

    pop_iqr = (meta[meta.model_group.isin([QWEN_GROUP, GEMINI_GROUP, GPT5_GROUP])]
               .groupby("model_group")["iqr"].mean())

    p1, p2, p3 = st.columns(3)
    p1.metric("6× Qwen baseline", f"{pop_iqr.get(QWEN_GROUP, float('nan')):.1f}",
              help=f"n = {(meta.model_group == QWEN_GROUP).sum()} runs")
    p2.metric("5× Qwen + Gemini", f"{pop_iqr.get(GEMINI_GROUP, float('nan')):.1f}",
              delta=f"+{pop_iqr.get(GEMINI_GROUP,0) - pop_iqr.get(QWEN_GROUP,0):.0f} vs baseline",
              help=f"n = {(meta.model_group == GEMINI_GROUP).sum()} runs")
    p3.metric("5× Qwen + GPT-5", f"{pop_iqr.get(GPT5_GROUP, float('nan')):.1f}",
              delta=f"{pop_iqr.get(GPT5_GROUP,0) - pop_iqr.get(QWEN_GROUP,0):+.0f} vs baseline",
              help=f"n = {(meta.model_group == GPT5_GROUP).sum()} runs")


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
