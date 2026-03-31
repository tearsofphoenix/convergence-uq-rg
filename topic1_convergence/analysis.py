"""
Topic 1: Neural Operator Convergence Theory
Convergence rate analysis and theory verification.

Compares empirical convergence rates to theoretical predictions:
  - For DeepONet: error ~ N^{-1/(2s)} where s is Sobolev regularity
  - For FNO: error ~ N^{-1/2} in L2 norm (empirical)
  - For PINNs: error depends on residual landscape

Generates plots and analysis tables.
"""
from __future__ import annotations
import numpy as np
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from pathlib import Path

@dataclass
class ConvergenceResult:
    pde_name: str
    solver_type: str  # "deeponet" | "fno" | "pinn"
    N_values: List[int]
    errors: List[float]
    fitted_alpha: float
    theory_alpha: float
    fitted_C: float  # constant in error = C * N^{-alpha}
    pde_dim: int
    network_config: Dict
    hardware: str
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def efficiency_ratio(self) -> float:
        """How close to theoretical rate (1.0 = perfect match)."""
        if self.theory_alpha == 0:
            return 0.0
        return self.fitted_alpha / self.theory_alpha

    def summary(self) -> str:
        return (f"{self.pde_name}/{self.solver_type}: "
                f"α_fit={self.fitted_alpha:.4f} vs α_theory={self.theory_alpha:.4f} "
                f"(ratio={self.efficiency_ratio():.2%})")


class ConvergenceAnalyzer:
    """
    Analyzes convergence rate experiments and generates reports.
    """
    def __init__(self, results_dir: str = "/Users/isaac/clawd/research/topic1_convergence/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[ConvergenceResult] = []

    def add(self, result: ConvergenceResult):
        self.results.append(result)

    def save(self, tag: str = "default"):
        path = self.results_dir / f"convergence_results_{tag}.json"
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        print(f"Saved {len(self.results)} results to {path}")

    def load(self, tag: str = "default"):
        path = self.results_dir / f"convergence_results_{tag}.json"
        if not path.exists():
            print(f"No results file found: {path}")
            return
        with open(path) as f:
            data = json.load(f)
        self.results = [ConvergenceResult(**d) for d in data]
        print(f"Loaded {len(self.results)} results")

    def summary_table(self) -> str:
        """Generate ASCII summary table."""
        lines = [
            "=" * 90,
            f"{'PDE':<25} {'Solver':<10} {'Dim':<4} {'α_fit':<8} {'α_theory':<9} {'Ratio':<8} {'Notes'}",
            "=" * 90,
        ]
        for r in sorted(self.results, key=lambda x: (x.pde_name, x.solver_type)):
            ratio = r.efficiency_ratio()
            ratio_str = f"{ratio:.1%}"
            status = "✓" if 0.7 <= ratio <= 1.3 else ("△" if ratio > 0.5 else "✗")
            lines.append(f"{r.pde_name:<25} {r.solver_type:<10} {r.pde_dim:<4} "
                        f"{r.fitted_alpha:<8.4f} {r.theory_alpha:<9.4f} {status} {ratio_str:<8} {r.notes}")
        lines.append("=" * 90)
        return "\n".join(lines)

    def fit_power_law(self, N_values: List[int], errors: List[float]) -> tuple:
        """
        Fit error(N) = C * N^{-alpha} via linear regression in log-log space.
        Returns: (alpha, log_C)
        """
        log_N = np.log(np.array(N_values, dtype=float))
        log_err = np.log(np.array(errors, dtype=float))
        alpha, log_C = np.polyfit(log_N, log_err, 1)
        return float(alpha), float(log_C)

    def theoretical_prediction(self, pde_name: str, solver_type: str, dim: int) -> float:
        """
        Return theoretical convergence rate based on PDE regularity and solver type.

        DeepONet: alpha ≈ s/d where s is Sobolev regularity and d is dimension
                  (from universal approximation theory)
        FNO: empirical rates typically 0.3-0.7 depending on PDE
        PINNs: highly problem-dependent, often 0.1-0.5
        """
        import sys
        sys.path.insert(0, "/Users/isaac/clawd/research/hermes")
        from shared.pde_suite import PDETestSuite
        suite = PDETestSuite()

        # Map display names to suite names
        name_map = {
            "poisson_2d": "poisson_2d_rect",
            "heat_1d": "heat_1d",
            "homogenization_2d": "homogenization_2d",
            "wave_1d": "wave_1d",
            "burgers_1d": "burgers_1d",
        }
        suite_name = name_map.get(pde_name, pde_name)

        try:
            case = suite.get(suite_name)
        except KeyError:
            # Fallback for unknown PDEs
            return 0.5

        if solver_type == "deeponet":
            # From theoretical analysis: alpha = s / d
            # Elliptic PDEs: s=2 (H^2 regularity) → alpha = 2/d
            # But empirical rates are typically lower due to network capacity
            s_map = {"low": 1.0, "medium": 1.5, "high": 2.0}
            s = s_map.get(case.difficulty, 1.5)
            return min(s / dim, 1.0) * 0.5  # empirical correction factor

        elif solver_type == "fno":
            # FNO typically achieves alpha ≈ 0.3-0.5 for smooth PDEs
            if case.pde_type.value == "elliptic":
                return 0.5
            elif case.pde_type.value == "parabolic":
                return 0.4
            else:
                return 0.3

        elif solver_type == "pinn":
            # PINNs are generally slower
            return 0.2

        return 0.5


if __name__ == "__main__":
    analyzer = ConvergenceAnalyzer()

    # Simulate some results for testing
    for pde in ["poisson_2d", "heat_1d", "homogenization_2d"]:
        for solver in ["deeponet", "fno"]:
            N_values = [100, 500, 1000, 5000]
            errors = [10 * n**-0.4 + np.random.randn()*0.1 for n in N_values]
            alpha, log_C = analyzer.fit_power_law(N_values, errors)
            theory = analyzer.theoretical_prediction(pde, solver, 2 if "2d" in pde else 1)

            result = ConvergenceResult(
                pde_name=pde,
                solver_type=solver,
                N_values=N_values,
                errors=errors,
                fitted_alpha=alpha,
                theory_alpha=theory,
                fitted_C=log_C,
                pde_dim=2 if "2d" in pde else 1,
                network_config={"modes": 16, "width": 64, "layers": 4},
                hardware="M4 Pro",
                notes="simulation"
            )
            analyzer.add(result)

    print(analyzer.summary_table())
    analyzer.save("simulation_test")
