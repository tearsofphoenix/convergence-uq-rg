"""
Generate legacy pilot ablation figure subplots individually + combined.

这些图对应 `ablation_rg.py` 的 3-seed pilot 输出，不属于当前论文主文
引用的统一 10-seed benchmark。
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_ROOT = Path("outputs/rg_bench/pilot_ablation")
OUT = OUT_ROOT / "figures"
SUBPLOT_OUT = OUT / "subplots"
SUBPLOT_OUT.mkdir(parents=True, exist_ok=True)

ABLATION_JSON = OUT_ROOT / "ablation_results.json"
if not ABLATION_JSON.exists():
    raise FileNotFoundError(
        f"Missing pilot ablation results: {ABLATION_JSON}. "
        "Run topic3_rg/ablation_rg.py first."
    )

with open(ABLATION_JSON) as f:
    data = json.load(f)

beta_c = 0.4407

def save_separate():
    """Save each subplot as a standalone tight-cropped PNG."""

    # ── A1: Block-Spin vs Random ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels = ["block-spin", "random"]
    means  = [data["ablation1"][k]["mean"] for k in labels]
    stds   = [data["ablation1"][k]["std"]  for k in labels]
    colors = ["#2171B5", "#E6550D"]
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    ax.set_ylabel("Test MSE")
    ax.set_title(f"Pilot A1: Block-Spin vs Random Labels\n(MLP, L=16, \u03b2={beta_c})")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.15,
                f"{m:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(SUBPLOT_OUT / "fig_ablations_a.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_ablations_a.png")

    # ── A2: MLP vs Linear ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels2 = ["MLP (3-layer)", "Linear (1-layer)"]
    means2  = [data["ablation2"][k]["mean"] for k in ["MLP", "Linear"]]
    stds2   = [data["ablation2"][k]["std"]  for k in ["MLP", "Linear"]]
    colors2 = ["#2171B5", "#6A51A3"]
    bars2 = ax.bar(labels2, means2, yerr=stds2, capsize=5, color=colors2, alpha=0.85)
    ax.set_ylabel("Test MSE")
    ax.set_title(f"Pilot A2: MLP vs Linear Baseline\n(L=16, \u03b2={beta_c})")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, m in zip(bars2, means2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.15,
                f"{m:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(SUBPLOT_OUT / "fig_ablations_b.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_ablations_b.png")

    # ── A3: Width robustness ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3.5))
    widths  = [64, 128, 256, 512]
    w_means = [data["ablation3"][str(w)]["mean"] for w in widths]
    w_stds  = [data["ablation3"][str(w)]["std"]  for w in widths]
    ax.bar([str(w) for w in widths], w_means, yerr=w_stds, capsize=4,
           color="#31A354", alpha=0.85)
    ax.set_xlabel("Hidden Width")
    ax.set_ylabel("Test MSE")
    ax.set_title(f"Pilot A3: Width Robustness\n(L=16, \u03b2={beta_c})")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    for i, (w, m) in enumerate(zip(widths, w_means)):
        ax.text(i, m*1.15, f"{m:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(SUBPLOT_OUT / "fig_ablations_c.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_ablations_c.png")

    # ── A4: Within-L baselines ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3.5))
    test_Ls = ["L4", "L8", "L16"]
    a4 = data["ablation4"]
    vals = [a4[L]["test_mse"] for L in test_Ls]
    x = np.arange(len(test_Ls))
    ax.bar(x, vals, 0.6, color="#2171B5", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(test_Ls)
    ax.set_xlabel("Lattice size")
    ax.set_ylabel("Test MSE")
    ax.set_title(f"Pilot A4: Block-Spin Learning by L\n(\u03b2={beta_c}, hidden=256)")
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(vals):
        ax.text(i, v*1.05, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(SUBPLOT_OUT / "fig_ablations_d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_ablations_d.png")

    # ── A5: Spectral radius vs β ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3.5))
    spec    = data["spectral_radius_corrected"]
    betas   = [r["beta"] for r in spec]
    rhos    = [r["rho"]   for r in spec]
    ax.axvline(beta_c, color="red", linestyle="--", lw=1.5, label=f"\u03b2c={beta_c}")
    ax.plot(betas, rhos, "bo-", ms=7, lw=2)
    ax.set_xlabel("\u03b2 (inverse temperature)")
    ax.set_ylabel("Spectral radius \u03c1 (max singular value)")
    ax.set_title("Pilot: Encoder Spectral Radius vs \u03b2\n(peak near criticality)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(SUBPLOT_OUT / "fig_ablations_e.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_ablations_e.png")

    # ── A6: Turbulence spectrum (placeholder) ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.text(0.5, 0.5,
            "Turbulence spectrum\n(k\u207b\u00b9\u00b3 \u00b1 0.05)\nRe=200, 2D LBM\n(k\u2208[2,20] inertial range)",
            ha="center", va="center", fontsize=11,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
    ax.set_title("Pilot: Turbulence Energy Spectrum\n(preliminary)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(SUBPLOT_OUT / "fig_ablations_f.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_ablations_f.png")


def save_combined():
    """Save the 2x3 combined figure (matches original layout)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Legacy Pilot Ablation Results (Not Main Benchmark)", fontsize=14)

    # A1
    ax = axes[0, 0]
    labels = ["block-spin", "random"]
    means  = [data["ablation1"][k]["mean"] for k in labels]
    stds   = [data["ablation1"][k]["std"]  for k in labels]
    colors = ["#2171B5", "#E6550D"]
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    ax.set_ylabel("Test MSE")
    ax.set_title(f"Pilot A1: Block-Spin vs Random Labels\n(MLP, L=16, \u03b2={beta_c})")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.1,
                f"{m:.4f}", ha="center", va="bottom", fontsize=9)

    # A2
    ax = axes[0, 1]
    labels2 = ["MLP (3-layer)", "Linear (1-layer)"]
    means2  = [data["ablation2"][k]["mean"] for k in ["MLP", "Linear"]]
    stds2   = [data["ablation2"][k]["std"]  for k in ["MLP", "Linear"]]
    colors2 = ["#2171B5", "#6A51A3"]
    bars2 = ax.bar(labels2, means2, yerr=stds2, capsize=5, color=colors2, alpha=0.85)
    ax.set_ylabel("Test MSE")
    ax.set_title(f"Pilot A2: MLP vs Linear Baseline\n(L=16, \u03b2={beta_c})")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, m in zip(bars2, means2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.1,
                f"{m:.4f}", ha="center", va="bottom", fontsize=9)

    # A3
    ax = axes[0, 2]
    widths  = [64, 128, 256, 512]
    w_means  = [data["ablation3"][str(w)]["mean"] for w in widths]
    w_stds   = [data["ablation3"][str(w)]["std"]  for w in widths]
    ax.bar([str(w) for w in widths], w_means, yerr=w_stds, capsize=4,
           color="#31A354", alpha=0.85)
    ax.set_xlabel("Hidden Width")
    ax.set_ylabel("Test MSE")
    ax.set_title(f"Pilot A3: Width Robustness\n(L=16, \u03b2={beta_c})")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    # A4
    ax = axes[1, 0]
    test_Ls = ["L4", "L8", "L16"]
    a4 = data["ablation4"]
    vals = [a4[L]["test_mse"] for L in test_Ls]
    x = np.arange(len(test_Ls))
    ax.bar(x, vals, 0.6, color="#2171B5", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(test_Ls)
    ax.set_xlabel("Lattice size")
    ax.set_ylabel("Test MSE")
    ax.set_title(f"Pilot A4: Block-Spin Learning by L\n(\u03b2={beta_c}, hidden=256)")
    ax.grid(True, alpha=0.3, axis="y")

    # A5: Spectral radius
    ax = axes[1, 1]
    spec    = data["spectral_radius_corrected"]
    betas   = [r["beta"] for r in spec]
    rhos    = [r["rho"]   for r in spec]
    ax.axvline(beta_c, color="red", linestyle="--", lw=1.5, label=f"\u03b2c={beta_c}")
    ax.plot(betas, rhos, "bo-", ms=7, lw=2)
    ax.set_xlabel("\u03b2 (inverse temperature)")
    ax.set_ylabel("Spectral radius \u03c1 (max singular value)")
    ax.set_title("Pilot: Encoder Spectral Radius vs \u03b2\n(peak near criticality)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # A6: Turbulence placeholder
    ax = axes[1, 2]
    ax.text(0.5, 0.5,
            "Turbulence spectrum\n(k\u207b\u00b9\u00b3 \u00b1 0.05)\nRe=200, 2D LBM\n(k\u2208[2,20] inertial range)",
            ha="center", va="center", fontsize=11,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
    ax.set_title("Pilot: Turbulence Energy Spectrum\n(preliminary)")
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(OUT / "fig_ablations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_ablations.png (combined)")


if __name__ == "__main__":
    save_separate()
    save_combined()
    print("\nAll figures saved.")
