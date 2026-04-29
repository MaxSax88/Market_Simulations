"""Pre-process the raw experiment pickle into compact parquet artifacts.

Reads `master_results_temp1_mem4.pkl` (a List[Dict] of 50-period market
experiments, ~700MB on disk because each dict carries a `messages` field of
LLM reasoning JSON), and writes two artifacts that the Streamlit dashboard
consumes:

  * dashboard_data.parquet — long format, one row per (run, time_step, agent),
    with predicted_price + actual_price.
  * runs_meta.parquet      — one row per run with precomputed price stats
    (peak, IQR, early/late std) used for filtering and badges.

Run once locally before deploying. The original pickle stays out of git.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd


def parse_seed(filename: str) -> int:
    stem = Path(filename).stem
    tail = stem.rsplit("_", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return -1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="master_results_temp1_mem4.pkl",
                    help="Path to the raw experiment pickle.")
    ap.add_argument("--output-dir", default=".",
                    help="Directory to write parquet artifacts into.")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {in_path} ...")
    with in_path.open("rb") as f:
        runs = pickle.load(f)
    print(f"Loaded {len(runs)} runs.")

    long_frames: list[pd.DataFrame] = []
    meta_rows: list[dict] = []

    for run_id, run in enumerate(runs):
        model_group = run["model_group"]
        temperature = run["temperature"]
        filename = run["filename"]
        seed = parse_seed(filename)

        df = run["results_df"][
            ["time_step", "agent_id", "predicted_price", "actual_price"]
        ].copy()
        df.insert(0, "run_id", run_id)
        df["model_group"] = model_group
        df["temperature"] = temperature
        df["seed"] = seed
        long_frames.append(df)

        anchor_agent = df["agent_id"].iloc[0]
        prices = (df.loc[df["agent_id"] == anchor_agent, ["time_step", "actual_price"]]
                  .sort_values("time_step"))
        ap_series = prices["actual_price"].astype(float).reset_index(drop=True)
        early = ap_series[prices["time_step"].reset_index(drop=True) < 25]
        late = ap_series[prices["time_step"].reset_index(drop=True) >= 25]

        meta_rows.append({
            "run_id": run_id,
            "model_group": model_group,
            "temperature": temperature,
            "seed": seed,
            "filename": filename,
            "peak_price": float(ap_series.max()),
            "min_price": float(ap_series.min()),
            "mean_price": float(ap_series.mean()),
            "iqr": float(ap_series.quantile(0.75) - ap_series.quantile(0.25)),
            "early_std": float(early.std(ddof=0)) if len(early) > 1 else 0.0,
            "late_std": float(late.std(ddof=0)) if len(late) > 1 else 0.0,
        })

    long_df = pd.concat(long_frames, ignore_index=True)
    long_df = long_df[
        ["run_id", "model_group", "temperature", "seed",
         "time_step", "agent_id", "predicted_price", "actual_price"]
    ]
    long_df["time_step"] = long_df["time_step"].astype("int16")
    long_df["agent_id"] = long_df["agent_id"].astype("int8")
    long_df["run_id"] = long_df["run_id"].astype("int32")
    long_df["predicted_price"] = long_df["predicted_price"].astype("float32")
    long_df["actual_price"] = long_df["actual_price"].astype("float32")

    meta_df = pd.DataFrame(meta_rows)

    data_path = out_dir / "dashboard_data.parquet"
    meta_path = out_dir / "runs_meta.parquet"
    long_df.to_parquet(data_path, index=False, compression="zstd")
    meta_df.to_parquet(meta_path, index=False, compression="zstd")

    print(f"\nWrote {data_path}  ({data_path.stat().st_size / 1e6:.1f} MB, {len(long_df):,} rows)")
    print(f"Wrote {meta_path}  ({meta_path.stat().st_size / 1e6:.1f} MB, {len(meta_df):,} rows)")

    print("\nRuns per model_group:")
    counts = meta_df.groupby("model_group").size().sort_values(ascending=False)
    for grp, n in counts.items():
        print(f"  {n:4d}  {grp}")

    GEMINI_GROUP = "5Qwen-Qwen3-14B_and_1gemini-3-flash-preview"
    GPT5_GROUP = "5Qwen-Qwen3-14B_and_1gpt-5-mini"
    QWEN_GROUP = "6Qwen-Qwen3-14B"

    cols = ["run_id", "seed", "filename", "peak_price",
            "iqr", "early_std", "late_std"]

    print("\n=== Page 3 hero candidates ===")

    for label, group, sort_col, ascending in [
        ("GEMINI — top peak_price (most dramatic bubble)", GEMINI_GROUP, "peak_price", False),
        ("GPT-5-mini — lowest IQR (flattest)", GPT5_GROUP, "iqr", True),
    ]:
        sub = (meta_df[meta_df["model_group"] == group]
               .sort_values(sort_col, ascending=ascending).head(8))
        print(f"\n{label}:")
        if sub.empty:
            print("  (no runs found)")
        else:
            print(sub[cols].to_string(index=False))

    qwen = meta_df[meta_df["model_group"] == QWEN_GROUP].copy()
    if not qwen.empty:
        median_iqr = qwen["iqr"].median()
        qwen["dist_from_median"] = (qwen["iqr"] - median_iqr).abs()
        sub = qwen.sort_values("dist_from_median").head(8)
        print(f"\n6× Qwen baseline — closest to median IQR ({median_iqr:.2f}):")
        print(sub[cols].to_string(index=False))


if __name__ == "__main__":
    main()
