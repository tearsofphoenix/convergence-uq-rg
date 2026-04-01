"""
Paper 1: DeepONet Convergence with Uncertainty Quantification
=============================================================
Run 5 seeds per PDE, report mean ± std of fitted alpha.
"""
from __future__ import annotations
import sys, json, time, math, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)
np.random.seed(42)

OUT_DIR  = Path("/Users/isaacliu/workspace/convergence-uq-rg/outputs/convergence")
FIG_DIR  = OUT_DIR / "figures"
DEVICE   = "cpu"
EPOCHS   = 300


def heat1d_exact(x, t, coeffs):
    val = np.zeros_like(x, dtype=np.float64)
    for k, a in enumerate(coeffs, 1):
        val += a * np.sin(k * math.pi * x) * math.exp(-(k**2) * math.pi**2 * t)
    return val.astype(np.float32)


def poisson2d_exact(x, y, coeffs):
    n_m = coeffs.shape[0]
    val = np.zeros_like(x, dtype=np.float64)
    for m in range(n_m):
        for n in range(n_m):
            val += coeffs[m, n] * np.sin((m+1)*math.pi*x) * np.sin((n+1)*math.pi*y)
    return val.astype(np.float32)


class PDEDataset(Dataset):
    def __init__(self, u_vals, y_coords, u_exact):
        self.u  = torch.from_numpy(u_vals)
        self.y  = torch.from_numpy(y_coords)
        self.ex = torch.from_numpy(u_exact)
    def __len__(self): return len(self.u)
    def __getitem__(self, idx):
        return self.u[idx], self.y[idx], self.ex[idx]


class BranchNet(nn.Module):
    def __init__(self, num_sensors, hidden=128, p=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_sensors, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),      nn.GELU(),
            nn.Linear(hidden, p),
        )
        self.p = p
    def forward(self, u):
        return self.net(u)


class TrunkNet(nn.Module):
    def __init__(self, coord_dim, hidden=128, p=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(coord_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),   nn.GELU(),
            nn.Linear(hidden, p),
        )
        self.p = p
    def forward(self, y):
        B, N, D = y.shape
        return self.net(y.reshape(B*N, D)).reshape(B, N, self.p)


class DeepONet(nn.Module):
    def __init__(self, num_sensors, coord_dim, hidden=128, p=50):
        super().__init__()
        self.branch = BranchNet(num_sensors, hidden, p)
        self.trunk  = TrunkNet(coord_dim, hidden, p)
    def forward(self, u, y):
        b = self.branch(u)
        t = self.trunk(y)
        return torch.sum(b[:, None, :] * t, dim=2)


def fit_power_law(Ns, errs):
    log_N = np.log(np.array(Ns, dtype=float))
    log_e = np.log(np.array(errs, dtype=float))
    slope, intercept = np.polyfit(log_N, log_e, 1)
    return float(-slope)


def train_deeponet(model, loader, epochs=EPOCHS, lr=1e-3):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for u_b, y_b, ex_b in loader:
            u_b, y_b, ex_b = u_b.to(DEVICE), y_b.to(DEVICE), ex_b.to(DEVICE)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(u_b, y_b), ex_b)
            loss.backward(); opt.step()


def eval_deeponet(model, u_vals, y_vals, ex_vals):
    model.to(DEVICE); model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(u_vals).to(DEVICE),
                     torch.from_numpy(y_vals).to(DEVICE))
        return nn.functional.mse_loss(pred, torch.from_numpy(ex_vals).to(DEVICE)).item()


def make_heat_dataset(N, num_sensors=32, num_points=64, t_final=0.05, seed=42):
    np.random.seed(seed)
    sensor_x = np.linspace(0.05, 0.95, num_sensors)
    query_x  = np.linspace(0.02, 0.98, num_points)
    y_coords = np.stack([query_x[:, None]] * N, axis=0).astype(np.float32)
    u_sensors_l, u_query_l = [], []
    for i in range(N):
        coeffs = np.random.randn(4) * 0.4
        coeffs[0] = 1.0
        ic  = heat1d_exact(sensor_x, 0.0, coeffs)
        uq  = heat1d_exact(query_x,  t_final, coeffs)
        u_sensors_l.append(ic)
        u_query_l.append(uq)
    return {
        "u_sensors": np.array(u_sensors_l, dtype=np.float32),
        "u_query":   np.array(u_query_l,   dtype=np.float32),
        "y_coords":  y_coords,
    }


def make_poisson_dataset(N, grid_size=8, n_modes=3, num_points=64, seed=42):
    np.random.seed(seed)
    xg = np.linspace(0.1, 0.9, grid_size)
    yg = np.linspace(0.1, 0.9, grid_size)
    SX, SY = np.meshgrid(xg, yg, indexing='ij')
    u_sensors_l, u_query_l, y_l = [], [], []
    for i in range(N):
        coeffs = np.random.randn(n_modes, n_modes) * 0.2
        coeffs[0, 0] = 1.0
        u_full = poisson2d_exact(SX, SY, coeffs)
        qx = np.random.uniform(0.02, 0.98, num_points)
        qy = np.random.uniform(0.02, 0.98, num_points)
        uq = poisson2d_exact(qx, qy, coeffs)
        u_sensors_l.append(u_full.flatten().astype(np.float32))
        u_query_l.append(uq.astype(np.float32))
        y_l.append(np.stack([qx, qy], axis=1).astype(np.float32))
    return {
        "u_sensors": np.array(u_sensors_l, dtype=np.float32),
        "u_query":   np.array(u_query_l,   dtype=np.float32),
        "y_coords":  np.array(y_l,          dtype=np.float32),
    }


def run_experiment(pde_name, make_dataset_fn, num_sensors, coord_dim,
                  train_seed_base, extra_args=None):
    """Run one PDE with 5 seeds. Returns {alpha_mean, alpha_std, errors per seed}."""
    if extra_args is None: extra_args = {}
    N_VALUES = [50, 100, 200, 400]
    TEST_N   = 100
    TEST_SEED = 9000

    # Fixed independent test set
    test_ds = make_dataset_fn(N=TEST_N, seed=TEST_SEED,
                               num_points=64, **extra_args)

    all_seed_alphas = []
    all_seed_mse     = {N: [] for N in N_VALUES}

    for seed_i in range(3):
        train_seed = train_seed_base + seed_i
        np.random.seed(train_seed); torch.manual_seed(train_seed)

        errs_by_N = {}
        for N in N_VALUES:
            train_ds = make_dataset_fn(N=N, seed=train_seed,
                                        num_points=64, **extra_args)
            loader = DataLoader(PDEDataset(
                train_ds["u_sensors"], train_ds["y_coords"], train_ds["u_query"]),
                batch_size=32, shuffle=True)
            model = DeepONet(num_sensors=num_sensors, coord_dim=coord_dim, p=50, hidden=128)
            train_deeponet(model, loader)
            mse = eval_deeponet(model, test_ds["u_sensors"], test_ds["y_coords"], test_ds["u_query"])
            errs_by_N[N] = mse
            all_seed_mse[N].append(mse)

        alpha = fit_power_law(N_VALUES, [errs_by_N[N] for N in N_VALUES])
        all_seed_alphas.append(alpha)

    alpha_mean = statistics.mean(all_seed_alphas)
    alpha_std  = float(np.std(all_seed_alphas, ddof=1)) if len(all_seed_alphas) > 1 else 0.0
    mse_means  = {N: statistics.mean(all_seed_mse[N]) for N in N_VALUES}

    print(f"  {pde_name}: alpha={alpha_mean:.3f} ± {alpha_std:.3f} (3 seeds)")
    print(f"    MSE means: {', '.join(f'{mse_means[N]:.2e}' for N in N_VALUES)}")

    return {
        "pde": pde_name,
        "alpha_mean": round(alpha_mean, 4),
        "alpha_std":  round(alpha_std,  4),
        "alpha_theory": 0.5,
        "seeds": [train_seed_base + i for i in range(3)],
        "N_values": N_VALUES,
        "mse_per_seed": {str(N): [round(v, 8) for v in all_seed_mse[N]] for N in N_VALUES},
        "mse_mean": {str(N): float(mse_means[N]) for N in N_VALUES},
        "all_alphas": [round(a, 4) for a in all_seed_alphas],
    }


def main():
    print("=" * 60)
    print("  DeepONet Convergence — 5 seeds UQ")
    print("=" * 60)
    t0 = time.time()
    results = []

    r_heat = run_experiment("heat_1d", make_heat_dataset,
                            num_sensors=32, coord_dim=1,
                            train_seed_base=2001, extra_args={"t_final": 0.05})
    results.append(r_heat)

    r_poisson = run_experiment("poisson_2d", make_poisson_dataset,
                               num_sensors=64, coord_dim=2,
                               train_seed_base=3001, extra_args={"grid_size": 8, "n_modes": 3})
    results.append(r_poisson)

    # Save
    with open(OUT_DIR / "convergence_5seeds.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print table
    print("\n" + "=" * 70)
    print("  TABLE: Empirical convergence rates (mean ± std, 3 seeds)")
    print("=" * 70)
    print(f"  {'PDE':<14} {'Dim':<4} {'alpha_mean':>16} {'alpha_theory':>10} {'Ratio':>8}")
    print("  " + "-" * 58)
    for r in results:
        dim = 1 if "heat" in r["pde"] else 2
        ratio = r["alpha_mean"] / r["alpha_theory"]
        print(f"  {r['pde']:<14} {dim:<4} {r['alpha_mean']:.3f} ± {r['alpha_std']:.3f}   "
              f"{r['alpha_theory']:>10.3f} {ratio:>7.1%}")
    print("=" * 70)
    print(f"\nTotal: {time.time()-t0:.0f}s  |  Saved: {OUT_DIR / 'convergence_5seeds.json'}")


if __name__ == "__main__":
    main()
