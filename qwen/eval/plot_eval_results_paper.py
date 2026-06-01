#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import re
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_RE = re.compile(r"Evaluation Report \| Method: ([A-Za-z0-9_]+)(?: \| Top-K: ([0-9]+))?")
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
    if not text:
        return None
    try:
        return dt.datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def parse_results(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    rows: List[Dict[str, object]] = []
    pending_time: Optional[str] = None
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

        run_match = RUN_RE.search(line)
        if run_match:
            current_method = run_match.group(1).upper()
            topk = run_match.group(2)
            current_topk = int(topk) if topk is not None else None
            current_row = None
            continue

        model_match = MODEL_RE.match(line)
        if model_match and current_method:
            model = model_match.group(1)
            current_row = {
                "time": pending_time,
                "time_dt": _parse_time(pending_time),
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

    return pd.DataFrame(rows)


def select_latest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if df["time_dt"].notna().any():
        df = df.sort_values(["time_dt"])
    latest = df.groupby(["method", "model"], as_index=False).tail(1)
    return latest


PALETTE = {
    "SYM": "#4BA6F6",
    "ASYM": "#4FE405",
    "FILIP": "#E5ED02",
}


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
    rest = [m for m in methods if m not in order]
    return [m for m in order if m in methods] + rest


def _model_order(models: List[str]) -> List[str]:
    order = ["SAE_V", "VL_SAE", "SAE_D"]
    rest = [m for m in models if m not in order]
    return [m for m in order if m in models] + rest


def grouped_bar(df: pd.DataFrame, value: str, title: str, ylabel: str, out_path: str, percent: bool = False):
    pivot = df.pivot(index="model", columns="method", values=value)
    methods = pivot.columns.tolist()
    models = pivot.index.tolist()
    x = np.arange(len(models))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, method in enumerate(methods):
        vals = pivot[method].values
        if percent:
            vals = vals * 100.0
        color = PALETTE.get(method, None)
        ax.bar(x + i * width, vals, width=width, label=method, color=color)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x + (len(methods) - 1) * width / 2)
    ax.set_xticklabels(models)
    ax.legend(frameon=False, ncol=min(3, len(methods)))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def radar_for_model(df: pd.DataFrame, model: str, out_path: str):
    subset = df[df["model"] == model]
    if subset.empty:
        return

    metrics = ["coverage", "align_score", "ev_mean", "primary"]
    labels = ["Coverage", "Align", "EV", "Primary"]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    for method in subset["method"].unique():
        row = subset[subset["method"] == method].iloc[0]
        values = [row[m] for m in metrics]
        values += values[:1]
        color = PALETTE.get(method, None)
        ax.plot(angles, values, label=method, linewidth=2, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(f"Radar: {model}")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", default="/home/liuzonghao/AASAE/qwen/eval/evaluation_results_all.txt")
    parser.add_argument("--out-dir", default="/home/liuzonghao/AASAE/qwen/eval/plots_paper")
    parser.add_argument("--methods", default="", help="Comma-separated filter")
    parser.add_argument("--models", default="", help="Comma-separated filter")
    args = parser.parse_args()

    df = parse_results(args.results_file)
    if df.empty:
        raise SystemExit("No Evaluation Report blocks found.")

    df = select_latest(df)
    if args.methods:
        keep = [m.strip().upper() for m in args.methods.split(",") if m.strip()]
        df = df[df["method"].isin(keep)]
    if args.models:
        keep = [m.strip().upper() for m in args.models.split(",") if m.strip()]
        df = df[df["model"].isin(keep)]

    if df.empty:
        raise SystemExit("No rows left after filtering.")

    df["ev_mean"] = (df["ev_v"] + df["ev_t"]) / 2.0
    df["dead_mean"] = (df["dead_v"] + df["dead_t"]) / 2.0
    df["r1_mean"] = (df["r1_i2t"] + df["r1_t2i"]) / 2.0
    df["r5_mean"] = (df["r5_i2t"] + df["r5_t2i"]) / 2.0
    df["align_score"] = (df["r1_mean"] + df["r5_mean"]) / 2.0

    method_order = _method_order(df["method"].unique().tolist())
    model_order = _model_order(df["model"].unique().tolist())
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df = df.sort_values(["model", "method"])

    os.makedirs(args.out_dir, exist_ok=True)
    _style()

    grouped_bar(df, "coverage", "Coverage (higher is better)", "Coverage", os.path.join(args.out_dir, "coverage.png"), percent=True)
    grouped_bar(df, "entail_ratio", "Entailment Ratio (lower is better)", "Entailment Ratio", os.path.join(args.out_dir, "entail_ratio.png"))
    grouped_bar(df, "ev_mean", "Mean EV (V/T)", "EV", os.path.join(args.out_dir, "ev_mean.png"))
    grouped_bar(df, "align_score", "Align Score (R@1/R@5)", "Align Score", os.path.join(args.out_dir, "align_score.png"))
    if "primary" in df.columns:
        grouped_bar(df, "primary", "Primary Score", "Primary Score", os.path.join(args.out_dir, "primary.png"))

    for model in model_order:
        radar_for_model(df, model, os.path.join(args.out_dir, f"radar_{model}.png"))

    df.sort_values(["method", "model"]).to_csv(os.path.join(args.out_dir, "paper_metrics.csv"), index=False)
    print(f"Saved paper plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
