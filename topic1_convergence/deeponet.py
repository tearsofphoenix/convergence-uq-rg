"""
Topic 1: Neural Operator Convergence Theory
DeepONet implementation in MLX for convergence rate experiments.

Architecture:
  - Branch Net: encodes the input function u(x) at sensor points
  - Trunk Net: evaluates at query location y
  - Output: G(u)(y) ≈ ∑_{i=1}^p b_i(y) * trunk_i(y)

Convergence Rate Experiment:
  - Vary number of training samples N (10^2 to 10^5)
  - Measure L2 error vs N
  - Fit power law: error ~ N^{-alpha}
  - Compare alpha vs theoretical predictions from Sobolev spaces
"""
from __future__ import annotations
import mlx.core as mx
import mlx.optimizers as opt
import mlx.nn as nn
from mlx.utils import tree_flatten
from typing import Tuple, Callable, Optional
import numpy as np

class BranchNet(nn.Module):
    """
    Encodes input function at sensor locations.
    Input shape: [batch, num_sensors, func_dim]
    Output shape: [batch, p] (p = number of basis functions)
    """
    def __init__(self, num_sensors: int, func_dim: int = 1, hidden_dims: list = [128, 128, 128], p: int = 100):
        super().__init__()
        dims = [num_sensors * func_dim] + hidden_dims + [p]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.branch = nn.Sequential(*layers)
        self.p = p

    def __call__(self, x: mx.array) -> mx.array:
        # x: [batch, num_sensors, func_dim]
        x = x.reshape(x.shape[0], -1)  # flatten sensors
        return self.branch(x)


class TrunkNet(nn.Module):
    """
    Evaluates trunk network at query location y.
    Input shape: [batch, num_points, coord_dim]
    Output shape: [batch, num_points, p]
    """
    def __init__(self, coord_dim: int = 2, hidden_dims: list = [128, 128, 128], p: int = 100):
        super().__init__()
        dims = [coord_dim] + hidden_dims + [p]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.trunk = nn.Sequential(*layers)
        self.p = p

    def __call__(self, y: mx.array) -> mx.array:
        # y: [batch, num_points, coord_dim]
        batch, num_pts, _ = y.shape
        y_flat = y.reshape(batch * num_pts, -1)
        out = self.trunk(y_flat)
        return out.reshape(batch, num_pts, self.p)


class DeepONet(nn.Module):
    """
    DeepONet architecture combining branch and trunk networks.
    G(u)(y) = ∑_{i=1}^p branch_i(u) * trunk_i(y)
    """
    def __init__(self, num_sensors: int, coord_dim: int = 2, p: int = 100,
                 branch_dims: list = None, trunk_dims: list = None):
        super().__init__()
        self.num_sensors = num_sensors
        self.coord_dim = coord_dim
        self.p = p
        self.branch = BranchNet(num_sensors, 1, branch_dims or [128, 128, 128], p)
        self.trunk = TrunkNet(coord_dim, trunk_dims or [128, 128, 128], p)

    def __call__(self, u: mx.array, y: mx.array) -> mx.array:
        """
        Args:
            u: [batch, num_sensors] — input function values at sensor locations
            y: [batch, num_points, coord_dim] — query coordinates
        Returns:
            [batch, num_points] — operator output at query points
        """
        b = self.branch(u)           # [batch, p]
        t = self.trunk(y)            # [batch, num_points, p]
        # Dot product over p dimension
        # b[:, None, :] * t[:, :, :]  → [batch, num_points, p]
        result = mx.sum(b[:, None, :] * t, axis=2)
        return result

    def count_parameters(self) -> int:
        return sum(p.size for _, p in tree_flatten(self.trainable_parameters()))


def pinn_loss(model: DeepONet, u: mx.array, y: mx.array,
              pde_residual_fn: Callable, BC_loss: mx.array = None) -> Tuple[mx.array, dict]:
    """
    Compute physics-informed loss for DeepONet (MLX functional version).

    Args:
        model: DeepONet
        u: [batch, num_sensors] — boundary/initial condition
        y: [batch, num_points, coord_dim] — interior query points
        pde_residual_fn: callable that computes PDE residual at (y, u(y))
        BC_loss: optional boundary condition loss term

    Returns:
        total_loss, dict of individual loss components
    """
    # Forward pass
    u_pred = model(u, y)

    # PDE residual (placeholder — specific to each PDE)
    # Note: pde_residual_fn should return an MX array, not numpy
    # grad_u placeholder: [batch, num_points, coord_dim] zeros
    grad_u_placeholder = mx.zeros((y.shape[0], y.shape[1], y.shape[2]))
    residual = pde_residual_fn(y, u_pred, grad_u_placeholder)

    # Data fidelity loss (if labels available)
    data_loss = mx.mean(u_pred ** 2)  # placeholder

    # BC loss
    bc = BC_loss if BC_loss is not None else 0.0

    loss = data_loss + bc + mx.mean(residual ** 2)

    return loss, {
        "data_loss": float(data_loss),
        "bc_loss": float(bc),
        "pde_residual": float(mx.mean(residual ** 2)),
        "total": float(loss)
    }


def generate_sensor_locations(num_sensors: int, domain: Tuple, scheme: str = "uniform") -> mx.array:
    """
    Generate sensor locations for function encoding.

    Args:
        num_sensors: number of sensor points
        domain: (xmin, xmax, ...) for each dimension
        scheme: "uniform" | "random" | "chebyshev"
    """
    dim = len(domain) // 2
    if scheme == "uniform":
        # Grid layout
        per_dim = int(round(num_sensors ** (1.0 / dim)))
        coords = [mx.linspace(domain[2*i], domain[2*i+1], per_dim) for i in range(dim)]
        grid = mx.meshgrid(*coords)
        sensors = mx.stack([g.flatten() for g in grid], axis=-1)
        return sensors[:num_sensors]
    elif scheme == "random":
        samples = []
        for i in range(dim):
            samples.append(mx.random.uniform(domain[2*i], domain[2*i+1], (num_sensors,)))
        return mx.stack(samples, axis=-1)
    elif scheme == "chebyshev":
        # Chebyshev nodes for better approximation
        import numpy as np
        per_dim = int(round(num_sensors ** (1.0 / dim)))
        nodes = np.cos(np.pi * (2 * np.arange(1, per_dim+1) - 1) / (2 * per_dim))
        nodes = (nodes + 1) / 2  # map to [0, 1]
        grid = np.meshgrid(*[nodes for _ in range(dim)])
        sensors = np.stack([g.flatten() for g in grid], axis=-1)
        return mx.array(sensors[:num_sensors])
    else:
        raise ValueError(f"Unknown scheme: {scheme}")


def convergence_experiment(
    pde_name: str,
    num_sensors: int = 100,
    p: int = 100,
    N_values: list = None,
    coord_dim: int = 2,
    device: str = "gpu"
) -> dict:
    """
    Run convergence rate experiment for DeepONet on a given PDE.

    Measures L2 error as a function of training samples N.
    Fits power law: error(N) ≈ C * N^{-alpha}
    Compares alpha to theoretical predictions.
    """
    if N_values is None:
        N_values = [100, 500, 1000, 5000, 10000]

    results = {
        "pde": pde_name,
        "N_values": N_values,
        "errors": [],
        "fitted_alpha": None,
        "theory_alpha": None,
    }

    # Set device
    if device == "gpu" and mx.default_device().kind != "gpu":
        mx.set_default_device(mx.gpu)
    else:
        mx.set_default_device(mx.cpu)

    for N in N_values:
        # Generate training data
        u_train = generate_random_inputs(N, num_sensors)
        y_train = generate_query_points(N, coord_dim, domain=get_domain(pde_name))

        # Initialize model
        model = DeepONet(num_sensors=num_sensors, coord_dim=coord_dim, p=p)
        optimizer = opt.Adam(learning_rate=1e-3)

        # Define loss function for MLX autograd (must take model as first arg)
        def compute_loss(m: DeepONet, u: mx.array, y: mx.array) -> mx.array:
            loss, _ = pinn_loss(m, u, y, get_pde_residual(pde_name))
            return loss

        # Train
        for epoch in range(2000):
            loss, _ = pinn_loss(model, u_train, y_train, get_pde_residual(pde_name))
            # MLX: use value_and_grad with argnums=[0] for model parameters
            loss_val, grads = mx.value_and_grad(compute_loss, argnums=0)(model, u_train, y_train)
            optimizer.update(model.parameters(), grads)

            if epoch % 500 == 0:
                print(f"  N={N}, epoch={epoch}, loss={float(loss):.6f}")

        # Evaluate on test set
        u_test = generate_random_inputs(1000, num_sensors)
        y_test = generate_query_points(1000, coord_dim, domain=get_domain(pde_name))
        u_exact_test = exact_solution(pde_name, y_test)

        u_pred_test = model(u_test, y_test)
        error = float(mx.mean((u_pred_test - u_exact_test) ** 2) ** 0.5)
        results["errors"].append(error)
        print(f"  → N={N}, L2 error = {error:.6e}")

    # Fit power law
    log_N = np.log(N_values)
    log_err = np.log(results["errors"])
    alpha, log_C = np.polyfit(log_N, log_err, 1)
    results["fitted_alpha"] = float(alpha)
    results["log_C"] = float(log_C)

    # Theory prediction
    results["theory_alpha"] = get_theoretical_alpha(pde_name)

    print(f"\nResults for {pde_name}:")
    print(f"  Fitted convergence rate α = {alpha:.4f}")
    print(f"  Theoretical rate α = {results['theory_alpha']:.4f}")

    return results


# === PDE-specific functions ===

def get_domain(pde_name: str) -> Tuple:
    domains = {
        "poisson_2d": (0.0, 1.0, 0.0, 1.0),
        "heat_1d": (0.0, 1.0),
        "wave_1d": (0.0, 1.0),
        "burgers_1d": (0.0, 1.0),
        "homogenization_2d": (0.0, 1.0, 0.0, 1.0),
    }
    return domains.get(pde_name, (0.0, 1.0))

def get_pde_residual(pde_name: str):
    """Return PDE-specific residual function. All return MX arrays."""
    residuals = {
        "poisson_2d": lambda y, u, grad_u: grad_u[:, :, 0] + grad_u[:, :, 1] + 2 * mx.pi**2 * mx.sin(mx.pi*y[:,:,0]) * mx.sin(mx.pi*y[:,:,1]),
        "heat_1d": lambda y, u, grad_u: grad_u[:, :, 0] - mx.pi**2 * u,
        "wave_1d": lambda y, u, grad_u: None,  # second-order in time — special handling
    }
    return residuals.get(pde_name, lambda y, u, g: mx.zeros_like(u))

def exact_solution(pde_name: str, y: mx.array) -> mx.array:
    """Get exact solution for validation."""
    if pde_name == "poisson_2d":
        return mx.sin(mx.pi * y[:, :, 0]) * mx.sin(mx.pi * y[:, :, 1])
    return mx.zeros((y.shape[0], y.shape[1]))

def get_theoretical_alpha(pde_name: str) -> float:
    """Theoretical convergence rate based on Sobolev space regularity."""
    rates = {
        "poisson_2d": 0.5,  # H^2 regularity → O(N^{-1/2}) for L2
        "heat_1d": 1.0,
        "wave_1d": 0.5,
        "burgers_1d": 0.25,
        "homogenization_2d": 0.25,
    }
    return rates.get(pde_name, 0.5)

def generate_random_inputs(N: int, num_sensors: int) -> mx.array:
    """Generate random input functions u(x) for training."""
    # Sample random coefficients in Fourier space
    coeffs = mx.random.normal((N, num_sensors // 2, 2))
    # Simple approach: random amplitude at each sensor
    return mx.random.uniform(-1.0, 1.0, (N, num_sensors))

def generate_query_points(N: int, coord_dim: int, domain: Tuple, num_points: int = 100) -> mx.array:
    """Generate query points y in the domain.

    Args:
        N: batch size
        coord_dim: spatial dimension
        domain: (xmin, xmax, ...) for each dimension
        num_points: number of query points per batch element
    """
    points = []
    for i in range(coord_dim):
        pts = mx.random.uniform(domain[2*i], domain[2*i+1], (N, num_points, 1))
        points.append(pts)
    return mx.concatenate(points, axis=-1)


if __name__ == "__main__":
    print("DeepONet convergence experiment framework")
    print(f"MLX device: {mx.default_device()}")
    print()

    # Quick test
    model = DeepONet(num_sensors=50, coord_dim=2, p=20)
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"  Branch output dim: {model.p}")
    print(f"  Trunk output dim: {model.p}")

    # Test forward pass
    u = mx.random.uniform(-1, 1, (4, 50))
    y = mx.random.uniform(0, 1, (4, 10, 2))
    out = model(u, y)
    print(f"  Forward pass: {u.shape} → {out.shape}")
