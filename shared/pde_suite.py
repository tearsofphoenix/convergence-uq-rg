"""
Shared PDE Test Suite — used by all 3 topics.
Canonical PDEs with analytical or high-precision reference solutions.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import abc

class PDEType(Enum):
    ELLIPTIC = "elliptic"
    PARABOLIC = "parabolic"
    HYPERBOLIC = "hyperbolic"
    MULTISCALE = "multiscale"
    HIGH_DIMENSIONAL = "high_dimensional"

@dataclass(frozen=True)
class PDECase:
    name: str
    pde_type: PDEType
    domain: Tuple[float, ...]
    dim: int
    has_analytical_solution: bool
    difficulty: str  # "low", "medium", "high"

    def __repr__(self):
        return f"PDECase({self.name}, dim={self.dim}, type={self.pde_type.value})"


class PDETestSuite:
    """
    Standardized benchmark suite for neural PDE solvers.
    Each case provides: solve(), exact_solution(), metadata.
    """

    def __init__(self):
        self.cases: Dict[str, PDECase] = {}
        self._register_all()

    def _register(self, case: PDECase):
        self.cases[case.name] = case

    def _register_all(self):
        # === ELLIPTIC ===
        self._register(PDECase(
            name="poisson_2d_lshape",
            pde_type=PDEType.ELLIPTIC,
            domain=(-1.0, 1.0, -1.0, 1.0),
            dim=2,
            has_analytical_solution=False,
            difficulty="medium",
        ))
        self._register(PDECase(
            name="poisson_2d_rect",
            pde_type=PDEType.ELLIPTIC,
            domain=(0.0, 1.0, 0.0, 1.0),
            dim=2,
            has_analytical_solution=True,
            difficulty="low",
        ))
        self._register(PDECase(
            name="poisson_3d_cube",
            pde_type=PDEType.ELLIPTIC,
            domain=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
            dim=3,
            has_analytical_solution=False,
            difficulty="high",
        ))
        self._register(PDECase(
            name="lamé_elasticity",
            pde_type=PDEType.ELLIPTIC,
            domain=(0.0, 1.0, 0.0, 1.0),
            dim=2,
            has_analytical_solution=False,
            difficulty="high",
        ))

        # === PARABOLIC ===
        self._register(PDECase(
            name="heat_1d",
            pde_type=PDEType.PARABOLIC,
            domain=(0.0, 1.0),
            dim=1,
            has_analytical_solution=True,
            difficulty="low",
        ))
        self._register(PDECase(
            name="heat_2d_square",
            pde_type=PDEType.PARABOLIC,
            domain=(0.0, 1.0, 0.0, 1.0),
            dim=2,
            has_analytical_solution=False,
            difficulty="medium",
        ))
        self._register(PDECase(
            name="reaction_diffusion",
            pde_type=PDEType.PARABOLIC,
            domain=(0.0, 1.0, 0.0, 1.0),
            dim=2,
            has_analytical_solution=False,
            difficulty="medium",
        ))

        # === HYPERBOLIC ===
        self._register(PDECase(
            name="wave_1d",
            pde_type=PDEType.HYPERBOLIC,
            domain=(0.0, 1.0),
            dim=1,
            has_analytical_solution=True,
            difficulty="low",
        ))
        self._register(PDECase(
            name="burgers_1d",
            pde_type=PDEType.HYPERBOLIC,
            domain=(0.0, 1.0),
            dim=1,
            has_analytical_solution=False,
            difficulty="high",
        ))
        self._register(PDECase(
            name="wave_2d_rect",
            pde_type=PDEType.HYPERBOLIC,
            domain=(0.0, 1.0, 0.0, 1.0),
            dim=2,
            has_analytical_solution=False,
            difficulty="medium",
        ))

        # === MULTISCALE ===
        self._register(PDECase(
            name="homogenization_2d",
            pde_type=PDEType.MULTISCALE,
            domain=(0.0, 1.0, 0.0, 1.0),
            dim=2,
            has_analytical_solution=False,
            difficulty="high",
        ))
        self._register(PDECase(
            name="navier_stokes_2d",
            pde_type=PDEType.MULTISCALE,
            domain=(0.0, 1.0, 0.0, 1.0),
            dim=2,
            has_analytical_solution=False,
            difficulty="very_high",
        ))

        # === HIGH DIMENSIONAL ===
        self._register(PDECase(
            name="high_dim_integration_100d",
            pde_type=PDEType.HIGH_DIMENSIONAL,
            domain=tuple([(0.0, 1.0)] * 10),  # actual dim=10, effective up to 100
            dim=10,
            has_analytical_solution=True,
            difficulty="high",
        ))

    def list_by_type(self, pde_type: PDEType) -> List[PDECase]:
        return [c for c in self.cases.values() if c.pde_type == pde_type]

    def get(self, name: str) -> PDECase:
        return self.cases[name]

    def summary(self) -> str:
        lines = ["PDE Test Suite Summary", "=" * 50]
        for ptype in PDEType:
            cases = self.list_by_type(ptype)
            lines.append(f"\n{ptype.value.upper()} ({len(cases)} cases):")
            for c in cases:
                lines.append(f"  - {c.name} (dim={c.dim}, exact={c.has_analytical_solution}, diff={c.difficulty})")
        return "\n".join(lines)


# === Analytical Solutions ===

def exact_poisson_2d_rect(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Analytical solution for Poisson equation on [0,1]x[0,1] with specific BCs."""
    # u(x,y) = sin(pi*x) * sin(pi*y), forcing f = 2*pi^2*sin(pi*x)*sin(pi*y)
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def exact_heat_1d(x: np.ndarray, t: float) -> np.ndarray:
    """Analytical solution for heat equation on [0,1] with homogeneous BCs."""
    # u(x,t) = sum_{n=1}^∞ 2*(1-(-1)^n)/(nπ) * sin(nπx) * exp(-(nπ)^2 * t)
    u = np.zeros_like(x)
    for n in range(1, 50):
        coeff = 2 * (1 - (-1)**n) / (n * np.pi)
        u += coeff * np.sin(n * np.pi * x) * np.exp(-(n * np.pi)**2 * t)
    return u

def exact_wave_1d(x: np.ndarray, t: float) -> np.ndarray:
    """D'Alembert solution for wave equation on [-1,1]."""
    # u(x,t) = (f(x-t) + f(x+t))/2, with f(x) = sin(pi*x)
    def f(x_val):
        return np.sin(np.pi * x_val)
    return (f(x - t) + f(x + t)) / 2

def exact_high_dim_integral(x: np.ndarray) -> np.ndarray:
    """
    Analytical integral of Gaussian: ∫ exp(-||x||^2) over unit hypercube.
    = (sqrt(pi)/2 * erf(1))^d for dimension d
    """
    from scipy.special import erf
    base = np.sqrt(np.pi) / 2 * erf(1)
    return base ** x.shape[-1] * np.ones(x.shape[:-1])


if __name__ == "__main__":
    suite = PDETestSuite()
    print(suite.summary())
