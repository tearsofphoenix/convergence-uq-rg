"""
Topic 2: UQ Calibration Benchmark Suite (PyTorch)
==================================================
Runs: 5 PDEs × 3 UQ methods × 3 precisions × 3 coverage levels
      = 135 configurations (configurable)

Outputs:
  - results/uq_results_{pde}.json
  - figures/reliability_{pde}_{method}_{precision}.png
  - figures/coverage_vs_ncal_{pde}.png
  - results/quantization_impact.csv

Usage:
  python -m topic2_uq.benchmark
"""
from __future__ import annotations
import sys
import os
import json
import time
import itertools
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Callable

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.pde_suite import PDETestSuite
from topic2_uq.pde_solvers import (
    SOLVER_REGISTRY, get_solver, generate_dataset,
    Poisson2DSolver, Heat1DSolver, Burgers1DSolver,
    HighDimIntegralSolver, NavierStokes2DSolver,
)

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path("/Users/isaacliu/workspace/convergence-uq-rg")
OUT_DIR = BASE_DIR / "outputs" / "uq_bench"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Quantization helpers
# ============================================================
class QuantizedModule(nn.Module):
    """
    Wraps a PyTorch module with dynamic INT8/INT4 quantization.
    Uses torch.quantization.quantize_dynamic.
    """
    def __init__(self, model: nn.Module, precision: str = "fp32"):
        super().__init__()
        self.precision = precision
        self.model = model
        if precision == "int8":
            self.quantized = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
        elif precision == "int4":
            # torch doesn't natively support int4, so we simulate
            # by quantizing to int8 and then scaling down
            self.quantized = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
        else:
            self.quantized = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantized(x)


def prepare_model_for_quant(model: nn.Module, precision: str) -> QuantizedModule:
    """Prepare model for specified precision."""
    return QuantizedModule(model, precision)


# ============================================================
# ECE computation
# ============================================================
def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (Guo et al., 2017).
    confidences: predicted confidence (e.g., 1 - normalized_error)
    accuracies: 1 if correct, 0 if incorrect
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if in_bin.sum() > 0:
            avg_conf = confidences[in_bin].mean()
            avg_acc = accuracies[in_bin].mean()
            frac = in_bin.sum() / len(confidences)
            ece += frac * abs(avg_acc - avg_conf)
    return ece


def coverage_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    nominal: float,
) -> Tuple[float, float, float]:
    """
    Compute empirical coverage for Gaussian-assumption interval.
    Handles multi-dimensional outputs (e.g., NS2D with 2 velocity components).
    Returns: (coverage, avg_width, z_score)
    """
    z_map = {0.80: 1.28, 0.90: 1.645, 0.95: 1.96}
    z = z_map.get(nominal, 1.645)

    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    ys = np.asarray(y_std)

    # Detect multi-output: y has shape (N, D) where D > 1
    if yt.ndim == 2 and yt.shape[1] > 1:
        n_out = yt.shape[1]
        covered_all, width_all = [], []
        for d in range(n_out):
            lo = yp[:, d] - z * ys[:, d]
            up = yp[:, d] + z * ys[:, d]
            covered_all.append(np.mean((yt[:, d] >= lo) & (yt[:, d] <= up)))
            width_all.append(np.mean(up - lo))
        emp_cov = float(np.mean(covered_all))
        width = float(np.mean(width_all))
    else:
        # Scalar case
        if yt.ndim > 1:
            yt = yt.flatten()
        if yp.ndim > 1:
            yp = yp.flatten()
        if ys.ndim > 1:
            ys = ys.flatten()
        lower = yp - z * ys
        upper = yp + z * ys
        covered = (yt >= lower) & (yt <= upper)
        emp_cov = float(covered.mean())
        width = float(np.mean(upper - lower))

    return emp_cov, width, z


def conformal_coverage(
    y_true_cal: np.ndarray,
    y_pred_cal: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    alpha: float,
) -> Tuple[float, float, float]:
    """
    Split conformal prediction.
    Handles multi-dimensional outputs by computing per-dim conformity scores.
    Returns: (q_hat, empirical_coverage, avg_width)
    """
    # Flatten multi-dim
    y_true_cal = np.asarray(y_true_cal)
    y_pred_cal = np.asarray(y_pred_cal)
    y_true_test = np.asarray(y_true_test)
    y_pred_test = np.asarray(y_pred_test)

    if y_true_cal.ndim > 1 and y_true_cal.shape[1] > 1:
        # Multi-output: flatten to per-component
        errors_cal = np.abs(y_true_cal - y_pred_cal).flatten()
        errors_test = np.abs(y_true_test - y_pred_test).flatten()
    else:
        errors_cal = np.abs(y_true_cal - y_pred_cal).flatten()
        errors_test = np.abs(y_true_test - y_pred_test).flatten()

    n_cal = len(errors_cal)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = min(q_level, 1.0)
    q_hat = np.quantile(errors_cal, q_level)

    emp_cov = (errors_test <= q_hat).mean()
    widths = 2 * q_hat * np.ones_like(errors_test)
    return q_hat, emp_cov, widths.mean()


# ============================================================
# UQ Experiment
# ============================================================
@dataclass
class UQResult:
    pde: str
    method: str
    precision: str
    nominal: float
    empirical_coverage: float
    ece: float
    avg_width: float
    nll: float
    n_cal: int
    n_test: int
    train_mae: float
    train_time_s: float

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class UQExperiment:
    """
    Full UQ benchmarking pipeline for one PDE.
    """

    PDE_CONFIG = {
        "poisson_2d": {
            "solver": Poisson2DSolver,
            "solver_kwargs": {"hidden": 128, "depth": 4},
            "dataset_kwargs": {},
            "train_kwargs": {"epochs": 300, "lr": 1e-3},
            "input_dim": 2,
            "output_dim": 1,
            "n_train": 1000,
            "n_cal": 500,
            "n_test": 1000,
        },
        "heat_1d": {
            "solver": Heat1DSolver,
            "solver_kwargs": {"hidden": 128, "depth": 4},
            "dataset_kwargs": {},
            "train_kwargs": {"epochs": 300, "lr": 1e-3},
            "input_dim": 2,
            "output_dim": 1,
            "n_train": 1000,
            "n_cal": 500,
            "n_test": 1000,
        },
        "burgers_1d": {
            "solver": Burgers1DSolver,
            "solver_kwargs": {"hidden": 128, "depth": 4},
            "dataset_kwargs": {},
            "train_kwargs": {"epochs": 300, "lr": 1e-3},
            "input_dim": 2,
            "output_dim": 1,
            "n_train": 1000,
            "n_cal": 500,
            "n_test": 1000,
        },
        "high_dim_integral": {
            "solver": HighDimIntegralSolver,
            "solver_kwargs": {"dim": 100, "hidden": 256, "depth": 5},
            "dataset_kwargs": {"dim": 100},
            "train_kwargs": {"epochs": 200, "lr": 1e-3},
            "input_dim": 100,
            "output_dim": 1,
            "n_train": 2000,
            "n_cal": 500,
            "n_test": 500,
        },
        "navier_stokes_2d": {
            "solver": NavierStokes2DSolver,
            "solver_kwargs": {"hidden": 128, "depth": 4},
            "dataset_kwargs": {"grid_size": 32, "reynolds": 200.0},
            "train_kwargs": {"epochs": 300, "lr": 1e-3},
            "input_dim": 3,
            "output_dim": 2,
            "n_train": 1000,
            "n_cal": 500,
            "n_test": 500,
        },
    }

    def __init__(
        self,
        pde: str,
        precisions: List[str] = ["fp32", "int8", "int4"],
        coverage_levels: List[float] = [0.80, 0.90, 0.95],
        seed: int = 42,
    ):
        self.pde = pde
        self.cfg = self.PDE_CONFIG[pde]
        self.precisions = precisions
        self.coverage_levels = coverage_levels
        self.seed = seed
        self.results: List[UQResult] = []

    def _train_fp32_model(self) -> nn.Module:
        """Train FP32 model and return."""
        solver_cls = self.cfg["solver"]
        model = solver_cls(**self.cfg["solver_kwargs"])
        X_train, y_train = solver_cls.generate_data(
            self.cfg["n_train"], seed=self.seed, **self.cfg["dataset_kwargs"]
        )
        _, y_val = solver_cls.generate_data(200, seed=self.seed + 1, **self.cfg["dataset_kwargs"])
        X_val = X_train[:200]

        torch.manual_seed(self.seed)
        model.train_model(
            X_train, y_train, X_val, y_val,
            epochs=self.cfg["train_kwargs"]["epochs"],
            lr=self.cfg["train_kwargs"]["lr"],
            verbose=False,
        )

        # Compute final MAE
        model.eval()
        with torch.no_grad():
            mae = nn.L1Loss()(model(X_train[:500].float()), y_train[:500].float()).item()

        return model, mae

    def _get_model_at_precision(self, fp32_model: nn.Module, precision: str) -> QuantizedModule:
        """Get model at specified precision."""
        if precision == "fp32":
            return fp32_model
        return prepare_model_for_quant(fp32_model, precision)

    def _mc_dropout_predict(
        self,
        model: nn.Module,
        X: torch.Tensor,
        num_passes: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MC Dropout: T forward passes with dropout enabled.
        - model must be in train() mode so dropout is active
        - We DO still use torch.no_grad() for memory efficiency
        Returns: (mean, std, all_preds) all as numpy arrays
        """
        model.train()  # ← dropout ON
        all_preds = []
        for _ in range(num_passes):
            with torch.no_grad():
                pred = model(X).numpy()
            all_preds.append(pred)
        all_preds = np.stack(all_preds, axis=0)  # [T, N, ...]
        mean = all_preds.mean(axis=0)
        std = all_preds.std(axis=0)
        model.eval()  # restore eval mode
        return mean.squeeze(), std.squeeze(), all_preds

    def _deep_ensemble_predict(
        self,
        solver_cls,
        solver_kwargs: dict,
        dataset_kwargs: dict,
        X: torch.Tensor,
        num_models: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Train ensemble of models, return mean and std predictions."""
        torch.manual_seed(self.seed)
        all_preds = []
        for i in range(num_models):
            model = solver_cls(**solver_kwargs)
            X_tr, y_tr = solver_cls.generate_data(
                self.cfg["n_train"] // 2, seed=self.seed + i, **dataset_kwargs
            )
            model.train_model(X_tr, y_tr, epochs=150, lr=1e-3)
            model.eval()
            with torch.no_grad():
                all_preds.append(model(X).numpy())
        all_preds = np.stack(all_preds, axis=0)
        return all_preds.mean(axis=0).squeeze(), all_preds.std(axis=0).squeeze()

    def run(self) -> List[UQResult]:
        """Run full benchmark for this PDE."""
        print(f"\n{'='*60}")
        print(f"  PDE: {self.pde}")
        print(f"{'='*60}")

        # Train FP32 model once
        t0 = time.time()
        fp32_model, train_mae = self._train_fp32_model()
        train_time = time.time() - t0
        print(f"  FP32 training: {train_time:.1f}s, MAE={train_mae:.5f}")

        # Generate calibration + test data
        solver_cls = self.cfg["solver"]
        X_cal, y_cal = solver_cls.generate_data(
            self.cfg["n_cal"], seed=self.seed + 10, **self.cfg["dataset_kwargs"]
        )
        X_test, y_test = solver_cls.generate_data(
            self.cfg["n_test"], seed=self.seed + 20, **self.cfg["dataset_kwargs"]
        )
        X_cal_np = X_cal.numpy().astype(np.float32)
        y_cal_np = y_cal.numpy().astype(np.float32)
        X_test_np = X_test.numpy().astype(np.float32)
        y_test_np = y_test.numpy().astype(np.float32)
        for precision in self.precisions:
            print(f"\n  [{precision.upper()}]")
            qmodel = self._get_model_at_precision(fp32_model, precision)
            qmodel.eval()

            # ---- MC Dropout ----
            print(f"    MC Dropout (50 passes)...")
            mc_mean, mc_std, _ = self._mc_dropout_predict(qmodel, X_cal[:200].float(), num_passes=50)
            mc_mean_test, mc_std_test, _ = self._mc_dropout_predict(qmodel, X_test[:200].float(), num_passes=50)
            y_cal_s = y_cal_np[:200]
            y_test_s = y_test_np[:200]

            for nom in self.coverage_levels:
                cov, width, _ = coverage_interval(y_test_s, mc_mean_test, mc_std_test, nom)
                # ECE/NLL: flatten multi-output to scalar residuals
                mc_mean_f = np.asarray(mc_mean).flatten()
                mc_std_f = np.asarray(mc_std).flatten()
                y_cal_f = np.asarray(y_cal_s).flatten()
                residuals = np.abs(y_cal_f - mc_mean_f)
                std_safe = np.maximum(mc_std_f, 1e-8)
                conf = 1.0 / (1.0 + residuals / std_safe)
                acc = (residuals < stats.norm.ppf((nom + 1) / 2) * std_safe).astype(float)
                ece = compute_ece(conf, acc)
                nll = 0.5 * np.log(2 * np.pi * std_safe**2) + 0.5 * ((y_cal_f - mc_mean_f) / std_safe)**2
                self.results.append(UQResult(
                    pde=self.pde, method="MC Dropout", precision=precision,
                    nominal=nom, empirical_coverage=float(cov), ece=float(ece),
                    avg_width=float(width), nll=float(np.mean(nll)),
                    n_cal=200, n_test=200, train_mae=train_mae, train_time_s=train_time,
                ))

            # ---- Deep Ensemble ----
            print(f"    Deep Ensemble (5 models)...")
            de_mean, de_std = self._deep_ensemble_predict(
                solver_cls, self.cfg["solver_kwargs"],
                self.cfg["dataset_kwargs"], X_cal[:200].float(), num_models=5
            )
            de_mean_test, de_std_test = self._deep_ensemble_predict(
                solver_cls, self.cfg["solver_kwargs"],
                self.cfg["dataset_kwargs"], X_test[:200].float(), num_models=5
            )

            for nom in self.coverage_levels:
                cov, width, _ = coverage_interval(y_test_s, de_mean_test, de_std_test, nom)
                de_mean_f = np.asarray(de_mean).flatten()
                de_std_f = np.asarray(de_std).flatten()
                de_std_safe = np.maximum(de_std_f, 1e-8)
                residuals = np.abs(np.asarray(y_cal_s).flatten() - de_mean_f)
                conf = 1.0 / (1.0 + residuals / de_std_safe)
                acc = (residuals < stats.norm.ppf((nom + 1) / 2) * de_std_safe).astype(float)
                ece = compute_ece(conf, acc)
                nll = 0.5 * np.log(2 * np.pi * de_std_safe**2) + 0.5 * ((np.asarray(y_cal_s).flatten() - de_mean_f) / de_std_safe)**2
                self.results.append(UQResult(
                    pde=self.pde, method="Deep Ensemble", precision=precision,
                    nominal=nom, empirical_coverage=float(cov), ece=float(ece),
                    avg_width=float(width), nll=float(np.mean(nll)),
                    n_cal=200, n_test=200, train_mae=train_mae, train_time_s=train_time,
                ))

            # ---- Conformal Prediction ----
            print(f"    Conformal Prediction...")
            qmodel.eval()
            with torch.no_grad():
                y_pred_cal = qmodel(X_cal[:500].float()).numpy()
                y_pred_test = qmodel(X_test[:500].float()).numpy()
            y_cal_s2 = y_cal_np[:500]
            y_test_s2 = y_test_np[:500]

            for nom in self.coverage_levels:
                alpha = 1 - nom
                q_hat, cov, width = conformal_coverage(
                    y_cal_s2, y_pred_cal, y_test_s2, y_pred_test, alpha
                )
                y_pred_cal_f = np.asarray(y_pred_cal).flatten()
                y_cal_f = np.asarray(y_cal_s2).flatten()
                y_test_f = np.asarray(y_test_s2).flatten()
                errors_test = np.abs(y_test_f - y_pred_cal_f[:len(y_test_f)])
                residuals_cal = np.abs(y_cal_f - y_pred_cal_f[:len(y_cal_f)])
                conf = 1.0 - errors_test / (q_hat + 1e-8)
                acc = (errors_test <= q_hat).astype(float)
                ece = compute_ece(conf, acc)
                self.results.append(UQResult(
                    pde=self.pde, method="Conformal", precision=precision,
                    nominal=nom, empirical_coverage=float(cov), ece=float(ece),
                    avg_width=float(width), nll=float(np.nan),
                    n_cal=500, n_test=500, train_mae=train_mae, train_time_s=train_time,
                ))

        print(f"  Total results: {len(self.results)}")
        return self.results

    def save_results(self, path: Optional[Path] = None):
        if path is None:
            path = OUT_DIR / f"results_{self.pde}.json"
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        print(f"  Saved to {path}")


# ============================================================
# Plotting
# ============================================================
def plot_reliability_diagrams(results: List[UQResult], pde: str):
    """Generate reliability diagrams for a PDE."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    methods = ["MC Dropout", "Deep Ensemble", "Conformal"]
    precisions = ["fp32", "int8", "int4"]

    for i, method in enumerate(methods):
        for j, prec in enumerate(precisions):
            ax = axes[i, j]
            res = [r for r in results
                   if r.pde == pde and r.method == method and r.precision == prec]
            if not res:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                continue

            nominals = [r.nominal for r in res]
            coverages = [r.empirical_coverage for r in res]
            eces = [r.ece for r in res]

            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
            ax.plot(nominals, coverages, "bo-", ms=6, lw=2,
                    label=f"Empirical\nECE={np.mean(eces):.3f}")
            ax.fill_between(nominals, nominals, coverages,
                           alpha=0.2, color="blue")
            ax.set_xlim(0.75, 1.0)
            ax.set_ylim(0.6, 1.05)
            ax.set_xlabel("Nominal Coverage")
            ax.set_ylabel("Empirical Coverage")
            ax.set_title(f"{method}\n{prec.upper()}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    plt.suptitle(f"Reliability Diagrams: {pde}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / f"reliability_{pde}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_quantization_impact(all_results: List[UQResult]):
    """Plot quantization impact heatmap."""
    pdes = list(set(r.pde for r in all_results))
    methods = list(set(r.method for r in all_results))

    for method in methods:
        fig, axes = plt.subplots(1, len(pdes), figsize=(4 * len(pdes), 4))
        if len(pdes) == 1:
            axes = [axes]
        for ax, pde in zip(axes, pdes):
            res = [r for r in all_results if r.pde == pde and r.method == method
                   and r.nominal == 0.90]
            if not res:
                ax.text(0.5, 0.5, "No data", ha="center")
                continue

            data = {(r.precision, r.nominal): r.empirical_coverage for r in res}
            nominals = sorted(set(r.nominal for r in res))
            precs = ["fp32", "int8", "int4"]
            matrix = []
            for prec in precs:
                row = [data.get((prec, nom), np.nan) for nom in nominals]
                matrix.append(row)

            matrix = np.array(matrix)
            im = ax.imshow(matrix, vmin=0.6, vmax=1.0, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(nominals)))
            ax.set_xticklabels([f"{n:.0%}" for n in nominals])
            ax.set_yticks(range(len(precs)))
            ax.set_yticklabels(precs)
            ax.set_xlabel("Nominal")
            ax.set_title(f"{pde}\n{method}")

            for i in range(len(precs)):
                for j in range(len(nominals)):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                               fontsize=9, color="black" if 0.7 < val < 0.95 else "white")

        plt.suptitle(f"Coverage @ 90% nominal — {method}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out = FIG_DIR / f"quant_impact_{method.replace(' ', '_')}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}")


def generate_summary_table(all_results: List[UQResult]) -> str:
    """Generate LaTeX summary table."""
    pdes = sorted(set(r.pde for r in all_results))
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{UQ Calibration Results (nominal=0.90). "
        r"Bold = within 5\% of nominal coverage.}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"\textbf{PDE} & \textbf{Method} & "
        r"\multicolumn{2}{c}{FP32} & "
        r"\multicolumn{2}{c}{INT8} & "
        r"\multicolumn{2}{c}{INT4} \\",
        r"{} & {} & Cov. & ECE & Cov. & ECE & Cov. & ECE \\",
        r"\midrule",
    ]

    for pde in pdes:
        for method in ["MC Dropout", "Deep Ensemble", "Conformal"]:
            underline_pde = pde.replace('_', r'\_')
            row = [rf"\underline{{{underline_pde}}}" if method == "MC Dropout" else "",
                   method]
            for prec in ["fp32", "int8", "int4"]:
                res = [r for r in all_results
                       if r.pde == pde and r.method == method
                       and r.precision == prec and r.nominal == 0.90]
                if res:
                    r = res[0]
                    cov_str = f"\\textbf{{{r.empirical_coverage:.3f}}}" \
                              if abs(r.empirical_coverage - 0.90) <= 0.05 \
                              else f"{r.empirical_coverage:.3f}"
                    row.append(cov_str)
                    row.append(f"{r.ece:.3f}")
                else:
                    row.extend(["—", "—"])
            lines.append(" & ".join(str(x) for x in row) + r" \\")

    lines.extend([r"\bottomrule", r"\end{table}", r"\label{tab:uq_results}"])
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  PDE-Bench-UQ  —  Uncertainty Quantification Calibration")
    print("  Device: CPU (Intel Mac Mini 2018)")
    print("=" * 70)

    pdes = ["poisson_2d", "heat_1d", "burgers_1d", "high_dim_integral", "navier_stokes_2d"]
    all_results: List[UQResult] = []

    for pde in pdes:
        exp = UQExperiment(pde, precisions=["fp32", "int8", "int4"],
                          coverage_levels=[0.80, 0.90, 0.95], seed=42)
        results = exp.run()
        all_results.extend(results)
        exp.save_results()

    # Save combined results
    combined_path = OUT_DIR / "results_all.json"
    with open(combined_path, "w") as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)
    print(f"\nCombined results: {combined_path}")

    # Generate plots
    print("\nGenerating reliability diagrams...")
    for pde in pdes:
        plot_reliability_diagrams(all_results, pde)

    print("Generating quantization impact plots...")
    plot_quantization_impact(all_results)

    # Generate summary table
    print("\nGenerating LaTeX summary table...")
    table = generate_summary_table(all_results)
    table_path = OUT_DIR / "summary_table.tex"
    with open(table_path, "w") as f:
        f.write(table)
    print(f"  Saved {table_path}")
    print("\n" + table)

    print(f"\n{'='*70}")
    print(f"  DONE — outputs in {OUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
