"""
Topic 2: Uncertainty Quantification Calibration for Neural PDE Solvers
UQ benchmarking suite — measures whether uncertainty estimates are honest.

Key question: Does a 90% prediction interval actually contain 90% of true values?

UQ Methods implemented:
  1. MC Dropout — variance over multiple forward passes with dropout
  2. Deep Ensemble — variance over multiple models with different init
  3. Conformal Prediction — distribution-free prediction sets

Calibration metrics:
  - Coverage: fraction of true values in predicted interval
  - ECE: Expected Calibration Error
  - Width: average prediction interval width
  - NLL: Negative Log-Likelihood

References:
  - Guo et al. (2017) "On Calibration of Modern Neural Networks"
  - Barber et al. (2023) "Conformal Prediction: A Unified Review"
  - Ovadia et al. (2019) "Can You Trust Your Model's Uncertainty?"
"""
from __future__ import annotations
from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

# Try to import MLX, fallback to numpy-only if not available
HAS_MLX = False
try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
    # Set MLX device - use .type instead of .kind
    try:
        if mx.default_device().type != mx.gpu:
            mx.set_default_device(mx.Device(mx.gpu))
    except Exception:
        pass  # Intel Mac without GPU or MLX not available
except ImportError:
    # MLX not available - will use numpy fallback
    mx = None
    nn = None


@dataclass
class UQExperimentResult:
    """Results from a single UQ calibration experiment."""
    method: str
    pde_name: str
    nominal_coverage: float      # e.g., 0.90
    empirical_coverage: float  # actual fraction inside interval
    ece: float                  # expected calibration error
    avg_interval_width: float
    nll: float
    num_test_samples: int

    def coverage_error(self) -> float:
        """Coverage gap: empirical - nominal. 0 is perfectly calibrated."""
        return self.empirical_coverage - self.nominal_coverage

    def is_calibrated(self, tol: float = 0.05) -> bool:
        return abs(self.coverage_error()) <= tol

    def summary(self) -> str:
        status = "CALIBRATED" if self.is_calibrated() else "MIS-CALIBRATED"
        return (f"{self.method:20s} | {self.pde_name:20s} | "
                f"nom={self.nominal_coverage:.2f} emp={self.empirical_coverage:.3f} "
                f"ECE={self.ece:.3f} width={self.avg_interval_width:.4f} | "
                f"{status}")


# Z-scores for normal approximation intervals at common coverage levels
Z_SCORES = {
    0.80: 1.28,
    0.90: 1.645,
    0.95: 1.96,
    0.99: 2.576,
}


class ConformalPredictor:
    """
    Conformal Prediction for neural PDE solvers.
    Provides distribution-free prediction sets with finite-sample coverage guarantees.

    For regression: predict set S(x) such that P(Y ∈ S(X)) ≥ 1-α (approximately)
    """
    def __init__(self, alpha: float = 0.1, method: str = "split"):
        """
        Args:
            alpha: miscoverage level (0.1 → 90% prediction sets)
            method: "split" (needs calibration set) or "full" (leave-one-out)
        """
        self.alpha = alpha
        self.method = method
        self.q_hat = None
        self.cal_scores = None

    def calibrate(self, errors: mx.array):
        """
        Compute quantile of errors on calibration set.
        errors: [N_cal] array of absolute residuals |y_true - y_pred|
        """
        err_np = np.array(errors)
        n = len(err_np)
        # Conformal quantile: ceil((n+1)*(1-α))/n
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self.q_hat = np.quantile(err_np, q_level)
        self.cal_scores = err_np
        return self.q_hat

    def predict(self, y_pred: mx.array, y_true: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        """
        Return prediction interval [y_pred - q_hat, y_pred + q_hat].
        If y_true provided, returns (lower, upper, covered) where covered is boolean.
        
        Raises:
            RuntimeError: if calibrate() has not been called first
        """
        if self.q_hat is None:
            raise RuntimeError(
                "ConformalPredictor must be calibrated before making predictions. "
                "Call calibrate(errors) first."
            )
        lower = y_pred - self.q_hat
        upper = y_pred + self.q_hat
        if y_true is not None:
            covered = mx.logical_and(y_true >= lower, y_true <= upper)
            return lower, upper, covered
        return lower, upper

    def coverage(self, lower: mx.array, upper: mx.array, y_true: mx.array) -> float:
        covered = mx.logical_and(y_true >= lower, y_true <= upper)
        return float(mx.mean(covered))


class MCDropoutUQ:
    """
    Monte Carlo Dropout for uncertainty estimation.
    Run T forward passes with dropout enabled, measure variance.
    """
    def __init__(self, model: nn.Module, num_passes: int = 50, dropout_rate: float = 0.1):
        self.model = model
        self.num_passes = num_passes
        self.dropout_rate = dropout_rate

    def predict(self, X: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Run T forward passes with dropout.
        Returns: (mean, std, all_predictions)
            mean: [N] mean prediction
            std:  [N] predictive std
            all_predictions: [T, N] all individual predictions
        """
        # MLX note: need to ensure dropout is active during eval
        # For simplicity, we use different random mask each pass
        predictions = []

        for _ in range(self.num_passes):
            # Apply random dropout mask
            mask = mx.random.bernoulli(1 - self.dropout_rate, X.shape)
            X_drop = X * mask / (1 - self.dropout_rate)
            pred = self.model(X_drop)
            predictions.append(pred)

        all_preds = mx.stack(predictions, axis=0)  # [T, N, ...]
        mean_pred = mx.mean(all_preds, axis=0)
        std_pred = mx.std(all_preds, axis=0)

        return mean_pred, std_pred, all_preds

    def calibration_metrics(self, X: mx.array, y_true: mx.array,
                           nominal_coverage: float = 0.90) -> UQExperimentResult:
        """Compute full calibration metrics for MC Dropout."""
        mean_pred, std_pred, all_preds = self.predict(X)

        # 90% interval: mean ± 1.645 * std (for Gaussian approx)
        z = 1.645  # for 90% interval
        lower = mean_pred - z * std_pred
        upper = mean_pred + z * std_pred

        # Empirical coverage
        covered = mx.logical_and(y_true >= lower, y_true <= upper)
        emp_cov = float(mx.mean(covered))

        # Interval width
        width = float(mx.mean(upper - lower))

        # ECE (Expected Calibration Error) with 10 bins
        ece = self._ece(std_pred, y_true, mean_pred, z_score=z, num_bins=10)

        # NLL
        nll = self._nll(mean_pred, std_pred, y_true)

        return UQExperimentResult(
            method="MC Dropout",
            pde_name="unknown",
            nominal_coverage=nominal_coverage,
            empirical_coverage=emp_cov,
            ece=ece,
            avg_interval_width=width,
            nll=nll,
            num_test_samples=len(y_true),
        )

    def _ece(self, std_pred, y_true, mean_pred, z_score: float = 1.645, num_bins: int = 10) -> float:
        """Expected Calibration Error.
        
        Args:
            std_pred: predicted standard deviations
            y_true: ground truth values
            mean_pred: predicted means
            z_score: z-score for the coverage level (default 1.645 for 90%)
            num_bins: number of bins for ECE calculation
            
        Returns:
            ECE: expected calibration error
        """
        residuals = mx.abs(y_true - mean_pred)
        avg_residual = float(mx.mean(residuals))
        avg_std = float(mx.mean(std_pred))
        # ECE = |mean residual - z * mean std| / z
        # If well-calibrated, residual ≈ z * std, so ECE ≈ 0
        if avg_std > 1e-8:
            return abs(avg_residual - z_score * avg_std) / z_score
        return avg_residual  # Fallback if std is near zero

    def _nll(self, mean_pred, std_pred, y_true) -> float:
        """Negative log-likelihood of Gaussian."""
        import numpy as np
        std_np = np.array(std_pred) + 1e-6
        mean_np = np.array(mean_pred)
        y_np = np.array(y_true)
        nll = 0.5 * np.log(2 * np.pi * std_np**2) + 0.5 * ((y_np - mean_np) / std_np)**2
        return float(np.mean(nll))


class DeepEnsembleUQ:
    """
    Deep Ensemble for uncertainty estimation.
    Train M models with different random seeds, measure variance.
    """
    def __init__(self, model_fn: Callable, num_models: int = 5, **model_kwargs):
        """
        model_fn: function that returns a new model instance
        """
        self.model_fn = model_fn
        self.num_models = num_models
        self.models = [model_fn() for _ in range(num_models)]
        self.kwargs = model_kwargs

    def predict(self, X: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Run all models, return mean, std, all predictions."""
        predictions = [model(X) for model in self.models]
        all_preds = mx.stack(predictions, axis=0)  # [M, N, ...]
        mean_pred = mx.mean(all_preds, axis=0)
        std_pred = mx.std(all_preds, axis=0)
        return mean_pred, std_pred, all_preds


class CalibrationReport:
    """
    Generates comprehensive UQ calibration reports.
    """
    def __init__(self):
        self.results: List[UQExperimentResult] = []

    def add(self, result: UQExperimentResult):
        self.results.append(result)

    def add_batch(self, results: List[UQExperimentResult]):
        self.results.extend(results)

    def summary(self) -> str:
        lines = [
            "=" * 100,
            "UQ CALIBRATION SUMMARY REPORT",
            "=" * 100,
            f"{'Method':<20} {'PDE':<20} {'Nominal':<8} {'Emp.Cov.':<10} {'ECE':<8} {'Width':<10} {'Status'}",
            "-" * 100,
        ]
        for r in self.results:
            lines.append(r.summary())
        lines.append("=" * 100)

        # Overall calibration rate
        n_calibrated = sum(1 for r in self.results if r.is_calibrated())
        lines.append(f"\nCalibration rate: {n_calibrated}/{len(self.results)} "
                    f"({n_calibrated/len(self.results)*100:.0f}%)")
        return "\n".join(lines)

    def save_json(self, path: str):
        import json
        data = [(type(r).__name__, r.__dict__) for r in self.results]
        # Convert to serializable format
        serializable = []
        for r in self.results:
            d = {k: float(v) if isinstance(v, (mx.array, np.floating)) else
                        (list(v) if isinstance(v, (list, mx.array)) else v)
                 for k, v in r.__dict__.items()}
            serializable.append(d)
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved {len(self.results)} results to {path}")


def run_uq_benchmark(
    pde_name: str,
    model_fn: Callable,
    num_test: int = 1000,
    coverage_levels: List[float] = None,
) -> List[UQExperimentResult]:
    """
    Run full UQ benchmark suite for a given PDE and model.

    Args:
        pde_name: name of PDE test case
        model_fn: function returning untrained model
        num_test: number of test samples
        coverage_levels: nominal coverage levels to test (default: [0.80, 0.90, 0.95])

    Returns:
        List of UQExperimentResult for each method × coverage level
    """
    if coverage_levels is None:
        coverage_levels = [0.80, 0.90, 0.95]

    # Generate test data (placeholder)
    # In practice: use actual PDE solver to get ground truth
    X_test = mx.random.normal((num_test, 100))
    y_test = mx.random.normal((num_test,))

    results = []
    report = CalibrationReport()

    # MC Dropout
    print(f"  Running MC Dropout...")
    mc_dropout = MCDropoutUQ(model_fn(), num_passes=50)
    for nom_cov in coverage_levels:
        mc_dropout.dropout_rate = 0.1 * (1 - nom_cov)  # adjust for coverage
        result = mc_dropout.calibration_metrics(X_test, y_test, nom_cov)
        result.pde_name = pde_name
        result.method = f"MC Dropout (α={nom_cov:.2f})"
        results.append(result)

    # Deep Ensemble
    print(f"  Running Deep Ensemble...")
    ensemble = DeepEnsembleUQ(model_fn, num_models=5)
    mean_pred, std_pred, _ = ensemble.predict(X_test)
    for nom_cov in coverage_levels:
        z = {0.80: 1.28, 0.90: 1.645, 0.95: 1.96}[nom_cov]
        lower = mean_pred - z * std_pred
        upper = mean_pred + z * std_pred
        covered = mx.logical_and(y_test >= lower, y_test <= upper)
        emp_cov = float(mx.mean(covered))
        width = float(mx.mean(upper - lower))

        result = UQExperimentResult(
            method=f"Deep Ensemble (α={nom_cov:.2f})",
            pde_name=pde_name,
            nominal_coverage=nom_cov,
            empirical_coverage=emp_cov,
            ece=abs(emp_cov - nom_cov),
            avg_interval_width=width,
            nll=0.0,
            num_test_samples=num_test,
        )
        results.append(result)

    # Conformal Prediction
    print(f"  Running Conformal Prediction...")
    model = model_fn()
    y_pred_cal = mx.random.normal((num_test,))  # placeholder
    errors_cal = mx.abs(y_test - y_pred_cal)
    cp = ConformalPredictor(alpha=1 - 0.90)
    cp.calibrate(errors_cal)
    y_pred_test = mx.random.normal((num_test,))
    lower, upper = cp.predict(y_pred_test)
    emp_cov = cp.coverage(lower, upper, y_test)
    width = float(mx.mean(upper - lower))

    results.append(UQExperimentResult(
        method="Conformal (90%)",
        pde_name=pde_name,
        nominal_coverage=0.90,
        empirical_coverage=emp_cov,
        ece=abs(emp_cov - 0.90),
        avg_interval_width=width,
        nll=0.0,
        num_test_samples=num_test,
    ))

    return results


if __name__ == "__main__":
    if not HAS_MLX:
        print("ERROR: MLX is required for this benchmark suite.")
        print("On Intel Mac without MLX support, please use a numpy-based UQ implementation.")
        exit(1)
        
    print("UQ Calibration Benchmark Suite")
    print(f"Device: {mx.default_device()}")

    # Quick sanity check
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(100, 1)
        def __call__(self, x):
            return self.fc(x)

    print("\nQuick MC Dropout test:")
    model = DummyModel()
    mc = MCDropoutUQ(model, num_passes=10)
    X = mx.random.normal((50, 100))
    mean_p, std_p, _ = mc.predict(X)
    print(f"  Input: {X.shape} → mean: {mean_p.shape}, std: {std_p.shape}")
    print(f"  Mean of std: {float(mx.mean(std_p)):.6f}")

    print("\nQuick benchmark simulation:")
    results = run_uq_benchmark("poisson_2d", DummyModel, num_test=200)
    report = CalibrationReport()
    report.add_batch(results)
    print(report.summary())
