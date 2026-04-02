"""
Generate summary plots for phase-2 assets:
  - XY nonlinear coarse-graining benchmark
  - Jacobian batch spectrum summaries
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(os.environ.get("PHASE2_PLOTS_OUT_DIR", "outputs/phase2_figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_xy_multiscale() -> None:
    data = load_json("outputs/xy_rg_multiscale/xy_rg_summary.json")
    betas = [0.60, 1.12, 1.50]
    models = ["Linear", "MLP", "CNN"]
    colors = {"Linear": "coral", "MLP": "steelblue", "CNN": "forestgreen"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex="col")
    for row, metric in enumerate(["mean_test_mse", "mean_abs_angle_error"]):
        for col, L in enumerate([8, 16]):
            ax = axes[row, col]
            for model in models:
                means = [data[f"L{L}_beta_{beta:.4f}_{model}"][metric] for beta in betas]
                std_key = metric.replace("mean_", "std_")
                stds = [data[f"L{L}_beta_{beta:.4f}_{model}"][std_key] for beta in betas]
                ax.plot(betas, means, marker="o", linewidth=2, color=colors[model], label=model)
                ax.fill_between(betas, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                                color=colors[model], alpha=0.15)
            ax.set_title(f"L={L}")
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_ylabel("Test MSE")
            else:
                ax.set_ylabel("Mean Abs Angle Error")
                ax.set_xlabel("beta")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("XY Nonlinear Coarse-Graining: Model Comparison Across Scale and Temperature", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT_DIR / "xy_multiscale_summary.png", dpi=160, bbox_inches="tight")
    plt.close()


def plot_jacobian_batch() -> None:
    cfgs = [
        ("L=8", load_json("outputs/rg_bench_wolff/jacobian_batch/summary.json")["summary"]),
        ("L=16", load_json("outputs/rg_bench_wolff/jacobian_batch_L16/summary.json")["summary"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    x = np.arange(5)
    for ax, (label, summary) in zip(axes, cfgs):
        for model, color in [("Linear", "coral"), ("MLP", "steelblue"), ("CNN", "forestgreen")]:
            means = np.array(summary[model]["top5_singular_mean"], dtype=float)
            stds = np.array(summary[model]["top5_singular_std"], dtype=float)
            ax.plot(x, means, marker="o", linewidth=2, color=color, label=model)
            ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.15)
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels([f"s{i+1}" for i in x])
        ax.set_xlabel("Singular value rank")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Jacobian singular value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Jacobian Spectrum Summary at beta_c (Wolff-sampled)", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(OUT_DIR / "jacobian_batch_summary.png", dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    plot_xy_multiscale()
    plot_jacobian_batch()
    print(OUT_DIR / "xy_multiscale_summary.png")
    print(OUT_DIR / "jacobian_batch_summary.png")


if __name__ == "__main__":
    main()
