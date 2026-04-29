# Machine Spirits — Interactive Dashboard

Companion dashboard to the paper *"Machine Spirits: Speculation and Adaptation
of LLM Agents in Asset Markets."* Six LLM agents trade a single asset over 50
periods; the price each period is determined by the average of their forecasts
plus a constant dividend, which makes optimism self-fulfilling. Different
models anchor very differently, and a single non-trend-following agent in a
mixed market can shift the entire dynamic.

The dashboard has three pages:

1. **The Spirit Gallery** — pick a homogeneous market (six copies of the same
   model) and watch how the realized price evolves vs each agent's forecast.
2. **Mixed Market Chaos** — a "🎲 Chaos Roulette" reveals how a fixed six-model
   mix produces wildly different outcomes purely from the random seed.
3. **The Adaptation Duel** — Gemini-3 Flash (volatility amplifier, "Apex
   Predator") vs GPT-5 mini (volatility damper) when each replaces one of six
   Qwen-3 14B trend-followers, anchored against the pure 6× Qwen baseline.

## Run locally

The dashboard reads two small parquet artifacts (`dashboard_data.parquet`,
`runs_meta.parquet`) produced once from the raw experiment pickle. The raw
pickle (~700 MB) is **not** committed; ask the authors for it.

```bash
git clone <this-repo>
cd Market_Simulations

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# One-time data preparation (requires master_results_temp1_mem4.pkl alongside).
python prepare_data.py --input /path/to/master_results_temp1_mem4.pkl

# Run the app.
streamlit run app.py
```

Open http://localhost:8501 in a browser.

## Deploy to Streamlit Community Cloud

1. Commit `app.py`, `requirements.txt`, `dashboard_data.parquet`, and
   `runs_meta.parquet`. **Do not** commit `master_results_temp1_mem4.pkl` —
   it's gitignored and exceeds GitHub's 100 MB limit.
2. Push to GitHub.
3. On https://share.streamlit.io, create a new app pointing at this repo and
   `app.py` as the entry point. No additional secrets or config needed.

The two parquet artifacts together are well under 1 MB, so the app boots
quickly and stays comfortably within the free tier's memory.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | The Streamlit dashboard. |
| `prepare_data.py` | One-shot ETL: pickle → parquet. |
| `dashboard_data.parquet` | Long-format prices: `(run_id, time_step, agent_id) → predicted_price, actual_price`. |
| `runs_meta.parquet` | One row per run with precomputed stats (peak, IQR, early/late std). |
| `requirements.txt` | Python dependencies. |

## Citation

> Anonymous. *Machine Spirits: Speculation and Adaptation of LLM Agents in
> Asset Markets.* 2026.
