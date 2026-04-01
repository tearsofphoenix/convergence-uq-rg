"""
Generate three figures for Paper 2:
1. Coverage vs Nominal (per PDE, per UQ method)
2. Width vs Coverage scatter
3. Delta-coverage by Precision bar chart
"""
import json, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path("/Users/isaacliu/workspace/convergence-uq-rg/outputs/uq_bench/figures")
OUT.mkdir(parents=True, exist_ok=True)

with open("/Users/isaacliu/workspace/convergence-uq-rg/outputs/uq_bench/results_all.json") as f:
    data = json.load(f)

# Filter: only Conformal and Deep Ensemble (MC Dropout is degenerate)
# and only FP32 for coverage-vs-nominal plot
METHODS  = ["Conformal", "Deep Ensemble"]
PDES     = ["poisson_2d", "heat_1d", "burgers_1d", "high_dim_integral", "navier_stokes_2d"]
PDE_LABELS = {
    "poisson_2d":        "Poisson 2D",
    "heat_1d":           "Heat 1D",
    "burgers_1d":        "Burgers 1D",
    "high_dim_integral": "100-d Integral",
    "navier_stokes_2d":  "Navier-Stokes 2D",
}
COLOR_MAP = {
    "Conformal":     "#2171B5",   # blue
    "Deep Ensemble": "#E6550D",   # orange
    "MC Dropout":    "#6A51A3",   # purple
}
MARKER_MAP = {
    "Conformal":     "o",
    "Deep Ensemble": "s",
    "MC Dropout":    "^",
}
PDE_MARKER = {
    "poisson_2d":        "o",
    "heat_1d":           "s",
    "burgers_1d":        "^",
    "high_dim_integral": "D",
    "navier_stokes_2d":  "P",
}
PDE_COLOR = {
    "poisson_2d":        "#1a9850",
    "heat_1d":           "#2166ac",
    "burgers_1d":        "#d73027",
    "high_dim_integral": "#fc8d59",
    "navier_stokes_2d":  "#7b3294",
}

# ===== PLOT 1: Coverage vs Nominal (FP32 only, conformal only) =====
def plot_coverage_vs_nominal():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Coverage vs. Nominal Level (FP32, Conformal Prediction)", fontsize=13, y=1.02)

    pdes_by_col = {
        0: ["poisson_2d", "heat_1d", "burgers_1d"],
        1: ["high_dim_integral", "navier_stokes_2d"],
        2: [],  # summary panel
    }

    # Subplot 0: elliptic/parabolic/hyperbolic
    ax = axes[0]
    for pde in pdes_by_col[0]:
        pts = [(r["nominal"], r["empirical_coverage"]) for r in data
               if r["pde"] == pde and r["method"] == "Conformal" and r["precision"] == "fp32"]
        pts.sort()
        nom, emp = zip(*pts)
        ax.plot(nom, emp, 'o-', color=PDE_COLOR[pde], label=PDE_LABELS[pde],
                linewidth=1.5, markersize=7)
    ax.plot([0.5, 1.0], [0.5, 1.0], 'k--', linewidth=1, label="Perfect calibration", alpha=0.6)
    ax.set_xlabel("Nominal Coverage", fontsize=10)
    ax.set_ylabel("Empirical Coverage", fontsize=10)
    ax.set_title("Low-dimensional PDEs", fontsize=10)
    ax.set_xlim(0.78, 0.97)
    ax.set_ylim(0.70, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Subplot 1: high-dim and turbulent
    ax = axes[1]
    for pde in pdes_by_col[1]:
        pts = [(r["nominal"], r["empirical_coverage"]) for r in data
               if r["pde"] == pde and r["method"] == "Conformal" and r["precision"] == "fp32"]
        pts.sort()
        nom, emp = zip(*pts)
        ax.plot(nom, emp, MARKER_MAP.get(pde, 'o') + '-', color=PDE_COLOR[pde],
                label=PDE_LABELS[pde], linewidth=1.5, markersize=7)
    ax.plot([0.5, 1.0], [0.5, 1.0], 'k--', linewidth=1, label="Perfect calibration", alpha=0.6)
    ax.set_xlabel("Nominal Coverage", fontsize=10)
    ax.set_ylabel("Empirical Coverage", fontsize=10)
    ax.set_title("High-dimensional / Turbulent", fontsize=10)
    ax.set_xlim(0.78, 0.97)
    ax.set_ylim(0.70, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Subplot 2: delta-coverage (empirical - nominal) per PDE
    ax = axes[2]
    pdes = ["poisson_2d", "heat_1d", "burgers_1d", "high_dim_integral", "navier_stokes_2d"]
    deltas = {}
    for pde in pdes:
        for nom in [0.8, 0.9, 0.95]:
            pts = [r["empirical_coverage"] - r["nominal"] for r in data
                   if r["pde"] == pde and r["method"] == "Conformal"
                   and r["precision"] == "fp32" and r["nominal"] == nom]
            if pts:
                deltas.setdefault(pde, []).append(pts[0])

    x = np.arange(3)
    width = 0.15
    for i, pde in enumerate(pdes):
        vals = deltas.get(pde, [0, 0, 0])
        ax.bar(x + i * width, vals, width, color=PDE_COLOR[pde],
               label=PDE_LABELS[pde], alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(["80%", "90%", "95%"])
    ax.set_xlabel("Nominal Coverage", fontsize=10)
    ax.set_ylabel("Coverage Gap (Empirical $-$ Nominal)", fontsize=10)
    ax.set_title("Calibration Gap by PDE", fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-0.15, 0.25)

    plt.tight_layout()
    path = OUT / "fig1_coverage_vs_nominal.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved {path}")
    plt.close()


# ===== PLOT 2: Width vs Coverage scatter (all precisions, conformal only) =====
def plot_width_vs_coverage():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Interval Width vs. Empirical Coverage (Conformal Prediction, all precisions)",
                 fontsize=13, y=1.02)

    precision_styles = {
        "fp32": {"marker": "o", "color": "#2171B5", "label": "FP32"},
        "int8": {"marker": "s", "color": "#E6550D", "label": "INT8"},
        "int4": {"marker": "^", "color": "#31A354", "label": "INT4"},
    }

    pdes_by_col = {
        0: ["poisson_2d", "heat_1d"],
        1: ["burgers_1d", "high_dim_integral"],
        2: ["navier_stokes_2d"],
    }

    for col_idx, pde_list in pdes_by_col.items():
        ax = axes[col_idx]
        for pde in pde_list:
            for prec, style in precision_styles.items():
                pts = [(r["empirical_coverage"], r["avg_width"])
                       for r in data
                       if r["pde"] == pde and r["method"] == "Conformal"
                       and r["precision"] == prec]
                if pts:
                    cov, w = zip(*pts)
                    ax.scatter(cov, w, marker=style["marker"], color=style["color"],
                               s=70, alpha=0.85,
                               label=f"{PDE_LABELS[pde]} / {style['label']}")

        ax.set_xlabel("Empirical Coverage", fontsize=10)
        ax.set_ylabel("Mean Interval Width", fontsize=10)
        ax.set_title(", ".join(PDE_LABELS[p] for p in pde_list), fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=1, loc='lower right')

    plt.tight_layout()
    path = OUT / "fig2_width_vs_coverage.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved {path}")
    plt.close()


# ===== PLOT 3: Delta-coverage by precision (bar chart) =====
def plot_delta_coverage_by_precision():
    pdes = ["poisson_2d", "heat_1d", "burgers_1d", "high_dim_integral", "navier_stokes_2d"]
    precisions = ["fp32", "int8", "int4"]
    prec_labels = {"fp32": "FP32", "int8": "INT8", "int4": "INT4"}
    nominals = [0.8, 0.9, 0.95]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Coverage Gap (Empirical $-$ Nominal) by Precision — Conformal Prediction",
                 fontsize=13, y=1.02)

    colors_by_pde = {
        "poisson_2d":        "#1a9850",
        "heat_1d":           "#2166ac",
        "burgers_1d":        "#d73027",
        "high_dim_integral": "#fc8d59",
        "navier_stokes_2d":  "#7b3294",
    }

    for ax_idx, nom in enumerate(nominals):
        ax = axes[ax_idx]
        x = np.arange(len(pdes))
        width = 0.25
        for i, prec in enumerate(precisions):
            vals = []
            for pde in pdes:
                gaps = [r["empirical_coverage"] - r["nominal"] for r in data
                        if r["pde"] == pde and r["method"] == "Conformal"
                        and r["precision"] == prec and r["nominal"] == nom]
                vals.append(gaps[0] if gaps else 0.0)
            bars = ax.bar(x + i * width, vals, width,
                          color=list(colors_by_pde.values())[i] if False else
                          ["#2171B5", "#E6550D", "#31A354"][i],
                          alpha=0.85,
                          label=prec_labels[prec])
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(x + width)
        ax.set_xticklabels([PDE_LABELS[p].replace(" ", "\n") for p in pdes],
                           fontsize=7.5)
        ax.set_ylabel("Coverage Gap", fontsize=9)
        ax.set_title(f"Nominal = {int(nom*100)}\%", fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(-0.18, 0.25)
        if ax_idx == 2:
            ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    path = OUT / "fig3_delta_coverage_by_precision.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved {path}")
    plt.close()


# ===== PLOT 4: Width vs Coverage for all 3 UQ methods (FP32, summary) =====
def plot_uq_method_comparison():
    """Scatter plot: empirical coverage vs interval width for all UQ methods, FP32."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Coverage vs. Width — Conformal vs Deep Ensemble (FP32)", fontsize=13, y=1.02)

    method_pairs = [("Conformal", "Deep Ensemble")]

    for ax_idx, method in enumerate(method_pairs):
        ax = axes[ax_idx]
        for pde in PDE_COLOR:
            pts = [(r["empirical_coverage"], r["avg_width"])
                   for r in data
                   if r["pde"] == pde and r["method"] == method[ax_idx]
                   and r["precision"] == "fp32"]
            if pts:
                cov, w = zip(*pts)
                ax.scatter(cov, w, marker=PDE_MARKER[pde], color=PDE_COLOR[pde],
                          s=80, alpha=0.85, label=PDE_LABELS[pde])

        ax.plot([0.5, 1.0], [0.5, 1.0], 'k--', linewidth=1, alpha=0.5, label="Perfect (y=x)")
        ax.set_xlabel("Empirical Coverage", fontsize=10)
        ax.set_ylabel("Mean Interval Width", fontsize=10)
        ax.set_title(f"{method[ax_idx]}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, ncol=2, loc='lower right')

    plt.tight_layout()
    path = OUT / "fig4_uq_method_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved {path}")
    plt.close()


if __name__ == "__main__":
    print("Generating figures...")
    plot_coverage_vs_nominal()
    plot_width_vs_coverage()
    plot_delta_coverage_by_precision()
    plot_uq_method_comparison()
    print("Done.")
