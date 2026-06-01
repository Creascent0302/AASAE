#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_HEADER_RE = re.compile(r"Evaluation Report \| Method: ([A-Za-z0-9_]+)(?: \| Top-K: ([0-9]+))?")
TIME_RE = re.compile(r"Test Time:\s*(.+)")
MODEL_RE = re.compile(r"^\[ (SAE_V|SAE_D|VL_SAE) \]$")

EV_RE = re.compile(r"EV \(V/T\): ([0-9.+-eE]+) / ([0-9.+-eE]+)")
DEAD_RE = re.compile(r"Dead Latents \(V/T\): ([0-9.+-eE]+)% / ([0-9.+-eE]+)%")
ALIGN_RE = re.compile(r"Align: AvgCos ([0-9.+-eE]+) \| PosSim ([0-9.+-eE]+)")
R1_RE = re.compile(r"Align R@1 \(I2T/T2I\): ([0-9.+-eE]+) / ([0-9.+-eE]+)")
R5_RE = re.compile(r"Align R@5 \(I2T/T2I\): ([0-9.+-eE]+) / ([0-9.+-eE]+)")
ENTAIL_RE = re.compile(r"Entail Ratio: ([0-9.+-eE]+) \| Coverage: ([0-9.+-eE]+)")
PRIMARY_RE = re.compile(r"Primary Score: ([0-9.+-eE]+)")


def _parse_time(text: str) -> Optional[dt.datetime]:
    try:
        return dt.datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def parse_results(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    runs: List[Dict[str, object]] = []
    rows: List[Dict[str, object]] = []

    pending_time: Optional[str] = None
    current_run_id: Optional[int] = None
    current_method: Optional[str] = None
    current_topk: Optional[int] = None
    current_row: Optional[Dict[str, object]] = None

    for line in lines:
        if not line:
            continue

        time_match = TIME_RE.search(line)
        if time_match:
            pending_time = time_match.group(1).strip()
            continue

        run_match = RUN_HEADER_RE.search(line)
        if run_match:
            method = run_match.group(1).upper()
            topk = run_match.group(2)
            current_run_id = len(runs)
            current_method = method
            current_topk = int(topk) if topk is not None else None
            runs.append(
                {
                    "run_id": current_run_id,
                    "time": pending_time,
                    "time_dt": _parse_time(pending_time) if pending_time else None,
                    "method": method,
                    "topk": current_topk,
                }
            )
            current_row = None
            continue

        model_match = MODEL_RE.match(line)
        if model_match and current_run_id is not None:
            model = model_match.group(1)
            current_row = {
                "run_id": current_run_id,
                "time": pending_time,
                "method": current_method,
                "topk": current_topk,
                "model": model,
            }
            rows.append(current_row)
            continue

        if current_row is None:
            continue

        m = EV_RE.search(line)
        if m:
            current_row["ev_v"] = float(m.group(1))
            current_row["ev_t"] = float(m.group(2))
            continue

        m = DEAD_RE.search(line)
        if m:
            current_row["dead_v"] = float(m.group(1))
            current_row["dead_t"] = float(m.group(2))
            continue

        m = ALIGN_RE.search(line)
        if m:
            current_row["align_avgcos"] = float(m.group(1))
            current_row["align_possim"] = float(m.group(2))
            continue

        m = R1_RE.search(line)
        if m:
            current_row["r1_i2t"] = float(m.group(1))
            current_row["r1_t2i"] = float(m.group(2))
            continue

        m = R5_RE.search(line)
        if m:
            current_row["r5_i2t"] = float(m.group(1))
            current_row["r5_t2i"] = float(m.group(2))
            continue

        m = ENTAIL_RE.search(line)
        if m:
            current_row["entail_ratio"] = float(m.group(1))
            current_row["coverage"] = float(m.group(2))
            continue

        m = PRIMARY_RE.search(line)
        if m:
            current_row["primary"] = float(m.group(1))
            continue

    runs_df = pd.DataFrame(runs)
    rows_df = pd.DataFrame(rows)
    return runs_df, rows_df


def select_runs(runs_df: pd.DataFrame, rows_df: pd.DataFrame, last_n: int) -> pd.DataFrame:
    if runs_df.empty or rows_df.empty:
        return rows_df

    runs_sorted = runs_df.copy()
    if runs_sorted["time_dt"].notna().any():
        runs_sorted = runs_sorted.sort_values(["time_dt", "run_id"])
    else:
        runs_sorted = runs_sorted.sort_values(["run_id"])

    if last_n > 0:
        keep = runs_sorted.tail(last_n)["run_id"].tolist()
    else:
        keep = runs_sorted.groupby("method", as_index=False).tail(1)["run_id"].tolist()

    return rows_df[rows_df["run_id"].isin(keep)].copy()


def _style():
    try:
        plt.style.use("seaborn-v0_8")
    except OSError:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        }
    )


def _method_order(methods: List[str]) -> List[str]:
    order = ["SYM", "ASYM", "FILIP"]
    remaining = [m for m in methods if m not in order]
    return [m for m in order if m in methods] + remaining


def _model_order(models: List[str]) -> List[str]:
    order = ["SAE_V", "VL_SAE", "SAE_D"]
    remaining = [m for m in models if m not in order]
    return [m for m in order if m in models] + remaining


def _agg_metrics(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c not in ("run_id", "time", "method", "topk", "model")]
    return df.groupby(["method", "model"], as_index=False)[numeric_cols].mean(numeric_only=True)


def _plot_grouped_bar(ax, pivot: pd.DataFrame, title: str, ylabel: str, percent: bool = False):
    methods = list(pivot.columns)
    models = list(pivot.index)
    x = np.arange(len(models))
    width = 0.8 / max(len(methods), 1)

    for i, method in enumerate(methods):
        vals = pivot[method].values
        if percent:
            vals = vals * 100.0
        ax.bar(x + i * width, vals, width=width, label=method)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x + (len(methods) - 1) * width / 2)
    ax.set_xticklabels(models)
    ax.legend(frameon=False, ncol=min(3, len(methods)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", default="/home/liuzonghao/AASAE/qwen/eval/evaluation_results_all.txt")
    parser.add_argument("--out-dir", default="/home/liuzonghao/AASAE/qwen/eval/plots")
    parser.add_argument("--last-n", type=int, default=0, help="Use last N runs; 0 means latest per method")
    parser.add_argument("--methods", default="", help="Comma-separated method filter")
    parser.add_argument("--models", default="", help="Comma-separated model filter")
    args = parser.parse_args()

    runs_df, rows_df = parse_results(args.results_file)
    if rows_df.empty:
        raise SystemExit("No 'Evaluation Report | Method:' blocks found in results file.")

    df = select_runs(runs_df, rows_df, args.last_n)

    if args.methods:
        keep_methods = [m.strip().upper() for m in args.methods.split(",") if m.strip()]
        df = df[df["method"].isin(keep_methods)]
    if args.models:
        keep_models = [m.strip().upper() for m in args.models.split(",") if m.strip()]
        df = df[df["model"].isin(keep_models)]

    if df.empty:
        raise SystemExit("No rows left after filtering.")

    os.makedirs(args.out_dir, exist_ok=True)

    df["ev_mean"] = (df["ev_v"] + df["ev_t"]) / 2.0
    df["dead_mean"] = (df["dead_v"] + df["dead_t"]) / 2.0
    df["r1_mean"] = (df["r1_i2t"] + df["r1_t2i"]) / 2.0
    df["r5_mean"] = (df["r5_i2t"] + df["r5_t2i"]) / 2.0
    df["align_score"] = (df["r1_mean"] + df["r5_mean"]) / 2.0

    agg = _agg_metrics(df)

    method_order = _method_order(agg["method"].unique().tolist())
    model_order = _model_order(agg["model"].unique().tolist())

    _style()

    # Coverage
    cov_pivot = agg.pivot(index="model", columns="method", values="coverage").reindex(index=model_order, columns=method_order)
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_grouped_bar(ax, cov_pivot, "Coverage (higher is better)", "Coverage", percent=True)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "coverage_by_method_model.png"))
    plt.close(fig)

    # Entailment ratio
    ent_pivot = agg.pivot(index="model", columns="method", values="entail_ratio").reindex(index=model_order, columns=method_order)
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_grouped_bar(ax, ent_pivot, "Entailment Ratio (lower is better)", "Entailment Ratio")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "entail_ratio_by_method_model.png"))
    plt.close(fig)

    # EV mean
    ev_pivot = agg.pivot(index="model", columns="method", values="ev_mean").reindex(index=model_order, columns=method_order)
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_grouped_bar(ax, ev_pivot, "Mean EV (V/T)", "EV")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "ev_mean_by_method_model.png"))
    plt.close(fig)

    # Dead latents
    dead_pivot = agg.pivot(index="model", columns="method", values="dead_mean").reindex(index=model_order, columns=method_order)
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_grouped_bar(ax, dead_pivot, "Dead Latents (lower is better)", "Dead Latents (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "dead_latents_by_method_model.png"))
    plt.close(fig)

    # Align score
    align_pivot = agg.pivot(index="model", columns="method", values="align_score").reindex(index=model_order, columns=method_order)
    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_grouped_bar(ax, align_pivot, "Align Score (mean of R@1/R@5)", "Align Score")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "align_score_by_method_model.png"))
    plt.close(fig)

    # Primary score
    if "primary" in agg.columns:
        prim_pivot = agg.pivot(index="model", columns="method", values="primary").reindex(index=model_order, columns=method_order)
        fig, ax = plt.subplots(figsize=(9, 5))
        _plot_grouped_bar(ax, prim_pivot, "Primary Score", "Primary Score")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "primary_score_by_method_model.png"))
        plt.close(fig)

    # Delta vs SYM (coverage, EV, align)
    if "SYM" in method_order and "ASYM" in method_order:
        delta_metrics = []
        for model in model_order:
            sym_row = agg[(agg["method"] == "SYM") & (agg["model"] == model)]
            asym_row = agg[(agg["method"] == "ASYM") & (agg["model"] == model)]
            if sym_row.empty or asym_row.empty:
                continue
            delta_metrics.append(
                {
                    "model": model,
                    "coverage_delta": float(asym_row["coverage"] - sym_row["coverage"]),
                    "ev_delta": float(asym_row["ev_mean"] - sym_row["ev_mean"]),
                    "align_delta": float(asym_row["align_score"] - sym_row["align_score"]),
                }
            )
        if delta_metrics:
            delta_df = pd.DataFrame(delta_metrics)
            fig, ax = plt.subplots(figsize=(9, 5))
            x = np.arange(len(delta_df["model"]))
            width = 0.25
            ax.bar(x - width, delta_df["coverage_delta"], width, label="Coverage")
            ax.bar(x, delta_df["ev_delta"], width, label="Mean EV")
            ax.bar(x + width, delta_df["align_delta"], width, label="Align Score")
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(delta_df["model"].tolist())
            ax.set_title("ASYM - SYM Delta")
            ax.set_ylabel("Delta")
            ax.legend(frameon=False, ncol=3)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, "delta_asym_vs_sym.png"))
            plt.close(fig)

    # Summary heatmap
    heat_cols = ["coverage", "ev_mean", "align_score", "dead_mean", "primary"]
    heat_cols = [c for c in heat_cols if c in agg.columns]
    if heat_cols:
        fig, axes = plt.subplots(1, len(heat_cols), figsize=(4 * len(heat_cols), 4), squeeze=False)
        for idx, col in enumerate(heat_cols):
            data = agg.pivot(index="model", columns="method", values=col).reindex(index=model_order, columns=method_order)
            ax = axes[0, idx]
            im = ax.imshow(data.values, aspect="auto", cmap="viridis")
            ax.set_xticks(np.arange(len(method_order)))
            ax.set_xticklabels(method_order)
            ax.set_yticks(np.arange(len(model_order)))
            ax.set_yticklabels(model_order)
            ax.set_title(col)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle("Metric Heatmaps")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "summary_heatmaps.png"))
        plt.close(fig)

    # Export table
    agg.sort_values(["method", "model"]).to_csv(os.path.join(args.out_dir, "metrics_table.csv"), index=False)
    print(f"Saved plots and table to: {args.out_dir}")


if __name__ == "__main__":
    main()
