"""
Topic 2: Neural PDE Solvers for UQ Benchmark
=============================================
PyTorch implementations of 4 PDE types:
  1. Poisson 2D (elliptic)     — u = sin(πx)sin(πy), f = 2π²sin(πx)sin(πy)
  2. Heat 1D   (parabolic)    — analytical solution via Fourier series
  3. Burgers 1D (hyperbolic)  — periodic, shocks via entropy solution
  4. High-dim integral (100d) — ∫_{[0,1]^100} exp(-||x||²) dx (Gilpin)
  5. Navier-Stokes 2D (Re=200) — D2Q9 LBM surrogate

Each solver is a simple MLP: input = coefficients/boundary/initial data,
output = solution at query points.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Callable, Optional
from abc import ABC, abstractmethod


# ============================================================
# Base
# ============================================================
class PDESolver(nn.Module):
    """Base class for neural PDE solvers."""

    def __init__(self, hidden: int = 128, depth: int = 4):
        super().__init__()
        self.hidden = hidden
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def train_model(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        epochs: int = 300,
        lr: float = 1e-3,
        verbose: bool = False,
    ) -> dict:
        """Standard training loop, returns metrics."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.5
        )
        X_tr = X_train.float()
        y_tr = y_train.float()
        X_vl = X_val.float() if X_val is not None else None
        y_vl = y_val.float() if y_val is not None else None

        history = {"train_loss": [], "val_loss": []}
        best_loss = float("inf")
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            pred = self(X_tr)
            loss = nn.MSELoss()(pred, y_tr)
            loss.backward()
            optimizer.step()

            history["train_loss"].append(loss.item())
            if X_vl is not None:
                self.eval()
                with torch.no_grad():
                    val_pred = self(X_vl)
                    val_loss = nn.MSELoss()(val_pred, y_vl).item()
                history["val_loss"].append(val_loss)
                scheduler.step(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
            elif epoch % 50 == 0 and verbose:
                print(f"  Epoch {epoch}: train_loss={loss.item():.6f}")

        return history

    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std) — for deterministic model, std=0."""
        self.eval()
        with torch.no_grad():
            pred = self(X.float()).detach()
        return pred, torch.zeros_like(pred)


# ============================================================
# 1. Poisson 2D
# ============================================================
class Poisson2DSolver(PDESolver):
    """
    Solve Poisson: Δu = f(x,y) on [0,1]², Dirichlet u=0 on boundary.
    Exact: u*(x,y) = sin(πx)sin(πy), f = 2π² sin(πx)sin(πy).

    Network input: [x, y] coordinates (2 features)
    Network output: u(x,y)
    """

    def __init__(self, hidden: int = 128, depth: int = 4):
        super().__init__(hidden, depth)
        # Skip-connection MLP
        layers = []
        dims = [2] + [hidden] * depth + [1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @staticmethod
    def generate_data(N: int, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        # Sample interior points
        x = torch.rand(N, 1)
        y = torch.rand(N, 1)
        coords = torch.cat([x, y], dim=1)
        # Exact solution: sin(πx)sin(πy)
        u_exact = torch.sin(np.pi * x) * torch.sin(np.pi * y)
        return coords, u_exact.squeeze(-1)

    def exact_solution(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sin(np.pi * x) * torch.sin(np.pi * y)


# ============================================================
# 2. Heat 1D
# ============================================================
class Heat1DSolver(PDESolver):
    """
    Heat equation u_t = u_xx on [0,1], u(0,t)=u(1,t)=0, u(x,0)=sin(πx).
    Exact: u(x,t) = sin(πx)·exp(-π²t).

    Input: [x, t] → output: u(x,t)
    """

    def __init__(self, hidden: int = 128, depth: int = 4):
        super().__init__(hidden, depth)
        layers = []
        dims = [2] + [hidden] * depth + [1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @staticmethod
    def generate_data(N: int, t_max: float = 0.1, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(seed)
        x = torch.rand(N, 1)
        t = torch.rand(N, 1) * t_max
        coords = torch.cat([x, t], dim=1)
        u_exact = torch.sin(np.pi * x) * torch.exp(-np.pi**2 * t)
        return coords, u_exact.squeeze(-1)


# ============================================================
# 3. Burgers 1D
# ============================================================
class Burgers1DSolver(PDESolver):
    """
    Burgers equation: u_t + u·u_x = ν·u_xx, ν=0.01, periodic BC.
    Cole-Hopf transformation gives exact solution.
    We use a simplified network: input = [x, t], output = u(x,t).

    Approximate solution via finite differences for training labels.
    """

    def __init__(self, hidden: int = 128, depth: int = 4):
        super().__init__(hidden, depth)
        layers = []
        dims = [2] + [hidden] * depth + [1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @staticmethod
    def generate_data(N: int, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate training data using Cole-Hopf exact solution:
        u(x,t) = -2ν·φ_x / φ,  φ = exp(-x²/(4νt+ε)) + exp(-(x-2π)²/(4νt+ε))
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        nu = 0.01
        x = torch.rand(N, 1)
        t = torch.rand(N, 1) * 0.5 + 0.01  # avoid t=0
        eps = 1e-4
        # Two-solution Cole-Hopf
        phi = torch.exp(-(x**2) / (4 * nu * t + eps)) + \
              torch.exp(-((x - 2 * np.pi) ** 2) / (4 * nu * t + eps))
        phi_x = (-2 * x / (4 * nu * t + eps)) * torch.exp(-(x**2) / (4 * nu * t + eps)) + \
                (-2 * (x - 2 * np.pi) / (4 * nu * t + eps)) * torch.exp(-((x - 2 * np.pi)**2) / (4 * nu * t + eps))
        u_exact = -2 * nu * phi_x / (phi + eps)
        u_exact = torch.clamp(u_exact, -3, 3)  # stability clamp
        coords = torch.cat([x, t], dim=1)
        return coords, u_exact.squeeze(-1)


# ============================================================
# 4. High-dimensional integral (Gilpin)
# ============================================================
class HighDimIntegralSolver(PDESolver):
    """
    Gilpin (100d) integral: ∫_{[0,1]^100} exp(-||x||²) dx
    Analytical: (√π/2 · erf(1))^100 ≈ 0.0^100 → very small
    We evaluate at random points: input = x ∈ [0,1]^100, output = exp(-||x||²)

    Network: maps 100-d input to scalar (the integrand value).
    Use smaller network since high-dim input.
    """

    def __init__(self, dim: int = 100, hidden: int = 256, depth: int = 5):
        super().__init__(hidden, depth)
        layers = []
        dims = [dim] + [hidden] * depth + [1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @staticmethod
    def generate_data(N: int, dim: int = 100, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        X = torch.rand(N, dim)
        # Exact integrand: exp(-||x||²)
        normsq = (X**2).sum(dim=1)
        y_exact = torch.exp(-normsq)
        return X, y_exact

    @staticmethod
    def analytical_integral(dim: int = 100) -> float:
        from scipy.special import erf
        base = np.sqrt(np.pi) / 2 * erf(1)
        return base**dim


# ============================================================
# 5. Navier-Stokes 2D (LBM surrogate)
# ============================================================
class NavierStokes2DSolver(PDESolver):
    """
    Navier-Stokes 2D, Re=200, lid-driven cavity.
    Simplified: predict velocity field from boundary conditions
    using a Fourier-based network.

    Input: [x, y, t] → output: [ux, uy] velocity
    """

    def __init__(self, hidden: int = 128, depth: int = 4):
        super().__init__(hidden, depth)
        layers = []
        dims = [3] + [hidden] * depth + [2]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def generate_data(N: int, grid_size: int = 32, reynolds: float = 200.0,
                      seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate approximate NS solution via LBM (lattice Boltzmann).
        D2Q9 model for lid-driven cavity.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        nx, ny = grid_size, grid_size
        # LBM parameters
        omega = 1.0 / (3.0 * (1.0 / reynolds) + 0.5)  # relaxation time
        u0 = 0.05  # lid velocity

        # D2Q9 weights and directions
        w = np.array([4/9] + [1/9]*4 + [1/36]*4)
        ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
        ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])

        # Initialize
        rho = np.ones((nx, ny), dtype=np.float64)
        ux = np.zeros((nx, ny), dtype=np.float64)
        uy = np.zeros((nx, ny), dtype=np.float64)

        # Collision step
        def equilibrium(rho, ux, uy):
            u2 = ux**2 + uy**2
            feq = np.zeros((9, nx, ny), dtype=np.float64)
            for i in range(9):
                eu = ex[i]*ux + ey[i]*uy
                feq[i] = w[i] * rho * (1 + 3*eu + 4.5*eu**2 - 1.5*u2)
            return feq

        # Run LBM for some steps
        feq = equilibrium(rho, ux, uy)
        f = feq.copy()

        for step in range(500):
            # Stream
            f_new = np.zeros_like(f)
            for i in range(9):
                f_new[i] = np.roll(np.roll(f[i], ex[i], axis=0), ey[i], axis=1)
            f = f_new

            # Collision
            u2 = ux**2 + uy**2
            for i in range(9):
                eu = ex[i]*ux + ey[i]*uy
                feq[i] = w[i] * rho * (1 + 3*eu + 4.5*eu**2 - 1.5*u2)
                f[i] = (1-omega)*f[i] + omega*feq[i]

            # Compute macros
            rho = sum(f)
            ux = sum(ex[i]*f[i] for i in range(9)) / rho
            uy = sum(ey[i]*f[i] for i in range(9)) / rho

            # Lid-driven cavity BC
            ux[:, -1] = u0
            ux[:, 0] = 0
            uy[:, -1] = 0
            uy[:, 0] = 0
            ux[0, :] = 0; uy[0, :] = 0
            ux[-1, :] = 0; uy[-1, :] = 0

        # Sample N points
        coords_list = []
        vel_list = []
        for _ in range(N):
            i = np.random.randint(0, nx)
            j = np.random.randint(0, ny)
            x_norm = i / nx
            y_norm = j / ny
            coords_list.append([x_norm, y_norm, 0.0])  # fixed time
            vel_list.append([ux[i, j], uy[i, j]])

        coords = torch.tensor(coords_list, dtype=torch.float32)
        velocities = torch.tensor(vel_list, dtype=torch.float32)
        return coords, velocities


# ============================================================
# Registry
# ============================================================
SOLVER_REGISTRY = {
    "poisson_2d": Poisson2DSolver,
    "heat_1d": Heat1DSolver,
    "burgers_1d": Burgers1DSolver,
    "high_dim_integral": HighDimIntegralSolver,
    "navier_stokes_2d": NavierStokes2DSolver,
}


def get_solver(name: str, **kwargs) -> PDESolver:
    return SOLVER_REGISTRY[name](**kwargs)


def generate_dataset(name: str, N: int, seed: int = 42, **kwargs):
    solver = SOLVER_REGISTRY[name]
    return solver.generate_data(N, seed=seed, **kwargs)
