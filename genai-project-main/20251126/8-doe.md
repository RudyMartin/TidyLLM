Got it — here’s a single, self-contained Python script that does the whole post-processing pipeline:

* loads `challenge_100_results.csv` and `nikoli_100_results.csv`
* aggregates metrics
* prints markdown + LaTeX tables to stdout
* generates a few key plots and saves them as PNGs

You can save this as `analyze_sudoku_results.py` and run it with `python analyze_sudoku_results.py`.

```python
#!/usr/bin/env python
"""
Analyze Sudoku-Bench experiment results for CTM/HRM and Bedrock transformers.

Expected inputs (CSV):
  results/challenge_100_results.csv
  results/nikoli_100_results.csv

Each CSV row corresponds to: (subset, puzzle_id, model_name, architecture, yrns_enabled, mode, solved, steps_taken, first_error_step, error_message).

Outputs:
  - Printed markdown and LaTeX tables
  - PNG plots saved under results/plots/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 0. Setup
# -----------------------------------------------------------------------------

def configure_matplotlib() -> None:
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["font.size"] = 11


# -----------------------------------------------------------------------------
# 1. Load Result Files
# -----------------------------------------------------------------------------

def load_results(results_dir: Path) -> pd.DataFrame:
    challenge_path = results_dir / "challenge_100_results.csv"
    nikoli_path = results_dir / "nikoli_100_results.csv"

    if not challenge_path.exists():
        raise FileNotFoundError(f"Missing file: {challenge_path}")
    if not nikoli_path.exists():
        raise FileNotFoundError(f"Missing file: {nikoli_path}")

    df_challenge = pd.read_csv(challenge_path)
    df_nikoli = pd.read_csv(nikoli_path)

    df_challenge["subset"] = "challenge_100"
    df_nikoli["subset"] = "nikoli_100"

    df = pd.concat([df_challenge, df_nikoli], ignore_index=True)
    return df


# -----------------------------------------------------------------------------
# 2. Basic Cleaning / Aggregation
# -----------------------------------------------------------------------------

def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["puzzle_id", "model_name", "mode"])

    # normalize types
    df["solved"] = df["solved"].astype(int)
    df["yrsn_enabled"] = df["yrsn_enabled"].astype(bool)

    # optional: ensure steps are ints
    if "steps_taken" in df.columns:
        df["steps_taken"] = df["steps_taken"].fillna(0).astype(int)
    if "first_error_step" in df.columns:
        df["first_error_step"] = df["first_error_step"].fillna(0).astype(int)

    return df


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df
        .groupby(["subset", "model_name", "architecture", "yrsn_enabled", "mode"], dropna=False)
        .agg(
            n_runs=("puzzle_id", "count"),
            n_solved=("solved", "sum"),
            solve_rate=("solved", "mean"),
            avg_steps=("steps_taken", lambda x: x.replace(0, np.nan).mean()),
            avg_first_error=("first_error_step", lambda x: x.replace(0, np.nan).mean()),
        )
        .reset_index()
    )
    grouped["solve_rate"] = 100 * grouped["solve_rate"]
    return grouped


# -----------------------------------------------------------------------------
# 3. Table Helpers (Markdown + LaTeX)
# -----------------------------------------------------------------------------

def df_to_markdown_table(df_table: pd.DataFrame) -> str:
    return df_table.to_markdown(index=False, floatfmt=".1f")


def df_to_latex_table(df_table: pd.DataFrame, caption: str, label: str) -> str:
    return df_table.to_latex(
        index=False,
        float_format="%.1f",
        caption=caption,
        label=label,
        escape=True,
    )


# -----------------------------------------------------------------------------
# 4. Plot Helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_solve_rates_vanilla_vs_yrsn(
    summary: pd.DataFrame,
    subset: str,
    mode: str,
    out_dir: Path,
) -> None:
    """
    Bar chart: solve rate per model, vanilla vs YRSN, for a given subset + mode.
    """
    sub = summary[(summary["subset"] == subset) & (summary["mode"] == mode)].copy()
    if sub.empty:
        print(f"[WARN] No data for subset={subset}, mode={mode}")
        return

    pivot = (
        sub
        .pivot_table(
            index=["model_name", "architecture"],
            columns="yrsn_enabled",
            values="solve_rate",
        )
        .rename(columns={False: "vanilla", True: "yrsn"})
        .reset_index()
    )

    pivot = pivot.sort_values("vanilla")
    x = np.arange(len(pivot))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, pivot["vanilla"], width, label="Vanilla")
    ax.bar(x + width / 2, pivot["yrsn"], width, label="YRSN")

    ax.set_xticks(x)
    ax.set_xticklabels(pivot["model_name"], rotation=45, ha="right")
    ax.set_ylabel("Solve rate (%)")
    ax.set_title(f"Solve rate ({subset}, {mode})")
    ax.legend()

    plt.tight_layout()

    ensure_dir(out_dir)
    out_path = out_dir / f"solve_rate_{subset}_{mode}.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved plot: {out_path}")


def plot_yrsn_delta(
    summary: pd.DataFrame,
    subset: str,
    mode: str,
    out_dir: Path,
) -> None:
    """
    Bar chart: improvement (YRSN - Vanilla) in solve rate, per model.
    """
    sub = summary[(summary["subset"] == subset) & (summary["mode"] == mode)].copy()
    if sub.empty:
        print(f"[WARN] No data for subset={subset}, mode={mode}")
        return

    pivot = (
        sub
        .pivot_table(
            index=["model_name", "architecture"],
            columns="yrsn_enabled",
            values="solve_rate",
        )
        .rename(columns={False: "vanilla", True: "yrsn"})
        .reset_index()
    )

    pivot["delta"] = pivot["yrsn"] - pivot["vanilla"]
    pivot = pivot.sort_values("delta")

    fig, ax = plt.subplots()
    ax.bar(pivot["model_name"], pivot["delta"])
    ax.axhline(0, linestyle="--", color="black", linewidth=1)
    ax.set_ylabel("Solve rate improvement (YRSN - Vanilla, %)")
    ax.set_title(f"Effect of YRSN on solve rate ({subset}, {mode})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    ensure_dir(out_dir)
    out_path = out_dir / f"yrsn_delta_{subset}_{mode}.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved plot: {out_path}")


# -----------------------------------------------------------------------------
# 5. Main analysis routine
# -----------------------------------------------------------------------------

def main() -> None:
    configure_matplotlib()

    results_dir = Path("results")
    plots_dir = results_dir / "plots"

    print(f"[INFO] Loading results from: {results_dir}")
    df_raw = load_results(results_dir)
    df = clean_results(df_raw)

    print("[INFO] Basic stats:")
    print(df["subset"].value_counts())
    print(df["model_name"].value_counts())
    print(df["mode"].value_counts())

    summary = aggregate_metrics(df)
    print("\n[INFO] Aggregated summary (head):")
    print(summary.head())

    # ---- RNN-only vs Transformer-only splits (if you want to inspect) ----
    summary_rnn = summary[summary["architecture"] == "rnn"].copy()
    summary_trf = summary[summary["architecture"] == "transformer"].copy()

    print("\n[INFO] RNN summary (head):")
    print(summary_rnn.head())

    print("\n[INFO] Transformer summary (head):")
    print(summary_trf.head())

    # ---- Example table: Single-shot, challenge_100 ----
    ss_challenge = summary[
        (summary["subset"] == "challenge_100") &
        (summary["mode"] == "single_shot")
    ].copy()

    if not ss_challenge.empty:
        table_ss_challenge = (
            ss_challenge
            .assign(
                YRSN=lambda d: np.where(d["yrsn_enabled"], "Yes", "No"),
                SolveRate=lambda d: d["solve_rate"].round(1),
            )[["model_name", "architecture", "YRSN", "SolveRate"]]
            .sort_values(["architecture", "model_name", "YRSN"])
        )

        print("\n[MARKDOWN] Single-shot, challenge_100:")
        print(df_to_markdown_table(table_ss_challenge))

        latex_ss = df_to_latex_table(
            table_ss_challenge.rename(columns={
                "model_name": "Model",
                "architecture": "Architecture",
                "YRSN": "YRSN",
                "SolveRate": "Solve rate (\\%)",
            }),
            caption="Single-shot performance on challenge\\_100.",
            label="tab:sudoku_single_shot_challenge",
        )
        print("\n[LATEX] Single-shot, challenge_100:")
        print(latex_ss)
    else:
        print("[WARN] No single-shot results for challenge_100 in summary.")

    # ---- Example table: Multi-step, challenge_100 ----
    ms_challenge = summary[
        (summary["subset"] == "challenge_100") &
        (summary["mode"] == "multi_step")
    ].copy()

    if not ms_challenge.empty:
        table_ms_challenge = (
            ms_challenge
            .assign(
                YRSN=lambda d: np.where(d["yrsn_enabled"], "Yes", "No"),
                SolveRate=lambda d: d["solve_rate"].round(1),
                AvgSteps=lambda d: d["avg_steps"].round(1),
                AvgFirstError=lambda d: d["avg_first_error"].round(1),
            )[["model_name", "architecture", "YRSN", "SolveRate", "AvgSteps", "AvgFirstError"]]
            .sort_values(["architecture", "model_name", "YRSN"])
        )

        print("\n[MARKDOWN] Multi-step, challenge_100:")
        print(df_to_markdown_table(table_ms_challenge))

        latex_ms = df_to_latex_table(
            table_ms_challenge.rename(columns={
                "model_name": "Model",
                "architecture": "Architecture",
                "YRSN": "YRSN",
                "SolveRate": "Solve rate (\\%)",
                "AvgSteps": "Avg steps",
                "AvgFirstError": "Avg first error step",
            }),
            caption="Multi-step performance on challenge\\_100.",
            label="tab:sudoku_multi_step_challenge",
        )
        print("\n[LATEX] Multi-step, challenge_100:")
        print(latex_ms)
    else:
        print("[WARN] No multi-step results for challenge_100 in summary.")

    # ---- Plots: solve rates and YRSN deltas ----
    for subset in ["challenge_100", "nikoli_100"]:
        for mode in ["single_shot", "multi_step"]:
            plot_solve_rates_vanilla_vs_yrsn(summary, subset=subset, mode=mode, out_dir=plots_dir)
            plot_yrsn_delta(summary, subset=subset, mode=mode, out_dir=plots_dir)

    print("\n[INFO] Analysis complete.")


if __name__ == "__main__":
    main()
```

You can trim or extend this as you start getting real data, but this gives you:

* one entry point
* tables you can paste into markdown / LaTeX
* PNG plots you can drop into slides or the dissertation.
