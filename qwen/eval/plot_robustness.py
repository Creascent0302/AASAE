#!/usr/bin/env python3
import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "SYM": "#5B8DB8",
    "ASYM": "#4CB5AE",
    "FILIP": "#B07AA1",
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


def plot_condition(df: pd.DataFrame, condition: str, out_dir: str):
    sub = df[df["condition"] == condition].copy()
    if sub.empty:
        return

    methods = _method_order(sub["method"].unique().tolist())
    models = _model_order(sub["model"].unique().tolist())

    metrics = [
        ("coverage", "Coverage (higher is better)"),
        ("entail_ratio", "Entailment Ratio (lower is better)"),
        ("align_score", "Align Score"),
        ("ev_mean", "EV Mean"),
        ("primary", "Primary Score"),
    ]

    for metric, title in metrics:
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True)
        if len(models) == 1:
            axes = [axes]

        for ax, model in zip(axes, models):
            mdf = sub[sub["model"] == model]
            for method in methods:
                sdf = mdf[mdf["method"] == method].sort_values("level")
                if sdf.empty:
                    continue
                color = PALETTE.get(method, None)
                ax.plot(sdf["level"], sdf[metric], marker="o", linewidth=2, label=method, color=color)
            ax.set_title(f"{model}")
            ax.set_xlabel("Level")
            ax.grid(True, linestyle="--", alpha=0.4)

        axes[0].set_ylabel(title)
        axes[0].legend(frameon=False, ncol=min(3, len(methods)))
        fig.suptitle(f"{title} vs {condition}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{condition}_{metric}.png"))
        plt.close(fig)

    if "SYM" in methods and "ASYM" in methods:
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True)
        if len(models) == 1:
            axes = [axes]
        for ax, model in zip(axes, models):
            mdf = sub[sub["model"] == model]
            sym = mdf[mdf["method"] == "SYM"].sort_values("level")
            asym = mdf[mdf["method"] == "ASYM"].sort_values("level")
            if sym.empty or asym.empty:
                continue
            levels = sym["level"].values
            delta = asym["coverage"].values - sym["coverage"].values
            ax.bar(levels, delta, width=0.03, color=PALETTE.get("ASYM", "#4CB5AE"))
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_title(f"{model}")
            ax.set_xlabel("Level")
            ax.grid(True, linestyle="--", alpha=0.4)
        axes[0].set_ylabel("Coverage Delta (ASYM - SYM)")
        fig.suptitle(f"Coverage Delta vs {condition}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{condition}_coverage_delta.png"))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/home/liuzonghao/AASAE/qwen/eval/robustness_metrics.csv")
    parser.add_argument("--out-dir", default="/home/liuzonghao/AASAE/qwen/eval/robustness_plots")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit("Empty robustness csv.")

    df["ev_mean"] = (df["ev_v"] + df["ev_t"]) / 2.0
    os.makedirs(args.out_dir, exist_ok=True)
    _style()

    for condition in df["condition"].unique():
        plot_condition(df, condition, args.out_dir)

    df.sort_values(["condition", "model", "method", "level"]).to_csv(
        os.path.join(args.out_dir, "robustness_table.csv"), index=False
    )
    print(f"Saved robustness plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
