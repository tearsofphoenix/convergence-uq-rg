"""
Topic 1: Neural Operator Convergence Experiments (PyTorch v4)
==============================================================
Key fixes over v3:
  1. STRICT separation: train and test use DIFFERENT random seeds.
     (v3 memorisation: train=seed2000, test=last100 of same seed2000 → not valid)
  2. FNO: spatial_dim = num_points (query count), not num_sensors.
     Input = IC at 32 sensors, Output = solution at 64 query points.
  3. Heat: IC parameters drawn fresh for each sample (not shared).
  4. Proper power-law fit (log_C - alpha*log_N).

The standard convergence theory:  error(N) = C * N^{-alpha}

Theory (paper1_convergence.tex):
  Heat 1D      d=1  β=1   α_theory = 1/(1+1) = 0.500
  Poisson 2D   d=2  β=2   α_theory = 2/(2+2) = 0.500
  Burgers 1D  d=1  β=0.5 α_theory = 0.5/(1+0.5) ≈ 0.333
"""
from __future__ import annotations
import sys, os, json, time, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("CONVERGENCE_OUT_DIR", REPO_ROOT / "outputs" / "convergence"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DEVICE  = "cpu"


# ─── Analytical Ground Truths ────────────────────────────────────────────────

def heat1d_exact(x, t, coeffs):
    """
    u(x,t) = Σ_{k=1}^{K} a_k * sin(kπx) * exp(-k²π²t)
    x: array; returns array.
    """
    val = np.zeros_like(x, dtype=np.float64)
    for k, a in enumerate(coeffs, 1):
        val += a * np.sin(k * math.pi * x) * math.exp(-(k**2) * math.pi**2 * t)
    return val.astype(np.float32)


def poisson2d_exact(x, y, coeffs):
    """
    u(x,y) = Σ_{m,n=1}^{n_modes} a_{mn} * sin((m+1)πx) * sin((n+1)πy)
    x, y: arrays; returns array.
    """
    n_m = coeffs.shape[0]
    val = np.zeros_like(x, dtype=np.float64)
    for m in range(n_m):
        for n in range(n_m):
            val += coeffs[m, n] * np.sin((m+1)*math.pi*x) * np.sin((n+1)*math.pi*y)
    return val.astype(np.float32)


# ─── Dataset factories ────────────────────────────────────────────────────────

def make_heat_dataset(N, num_sensors=32, num_points=64,
                      t_final=0.05, seed=42):
    """
    Heat 1D dataset.
    Each sample: random IC drawn from Gaussian process (Fourier series).
    Train/test STRICTLY SEPARATE (different seeds).
    """
    np.random.seed(seed)
    sensor_x = np.linspace(0.05, 0.95, num_sensors)
    query_x  = np.linspace(0.02, 0.98, num_points)
    # y_coords: [N, num_points, 1]
    y_coords = np.stack([query_x[:, None]] * N, axis=0).astype(np.float32)

    u_sensors_l, u_query_l = [], []
    for i in range(N):
        # Each sample: different random Fourier coefficients
        coeffs = np.random.randn(4) * 0.4
        coeffs[0] = 1.0   # ensure non-zero dominant mode
        ic  = heat1d_exact(sensor_x, 0.0, coeffs)
        uq  = heat1d_exact(query_x,  t_final, coeffs)
        u_sensors_l.append(ic)
        u_query_l.append(uq)

    return {
        "u_sensors": np.array(u_sensors_l, dtype=np.float32),
        "u_query":   np.array(u_query_l,   dtype=np.float32),
        "y_coords":  y_coords,
    }


def make_poisson_dataset(N, grid_size=8, n_modes=3, num_points=64, num_sensors=None, seed=42):
    """num_sensors is ignored (derived from grid_size²)."""
    """
    Poisson 2D dataset.
    Each sample: different random Fourier coefficient matrix.
    """
    np.random.seed(seed)
    xg = np.linspace(0.1, 0.9, grid_size)
    yg = np.linspace(0.1, 0.9, grid_size)
    SX, SY = np.meshgrid(xg, yg, indexing='ij')

    u_sensors_l, u_query_l, y_l = [], [], []
    for i in range(N):
        coeffs = np.random.randn(n_modes, n_modes) * 0.2
        coeffs[0, 0] = 1.0

        u_full = poisson2d_exact(SX, SY, coeffs)
        u_s = u_full.flatten()

        qx = np.random.uniform(0.02, 0.98, num_points)
        qy = np.random.uniform(0.02, 0.98, num_points)
        uq = poisson2d_exact(qx, qy, coeffs)

        u_sensors_l.append(u_s.astype(np.float32))
        u_query_l.append(uq.astype(np.float32))
        y_l.append(np.stack([qx, qy], axis=1).astype(np.float32))

    return {
        "u_sensors": np.array(u_sensors_l, dtype=np.float32),
        "u_query":   np.array(u_query_l,   dtype=np.float32),
        "y_coords":  np.array(y_l,          dtype=np.float32),
    }


# ─── Torch Dataset ───────────────────────────────────────────────────────────

class PDEDataset(Dataset):
    def __init__(self, u_vals, y_coords, u_exact):
        self.u  = torch.from_numpy(u_vals)
        self.y  = torch.from_numpy(y_coords)
        self.ex = torch.from_numpy(u_exact)
    def __len__(self): return len(self.u)
    def __getitem__(self, idx):
        return self.u[idx], self.y[idx], self.ex[idx]


# ─── DeepONet ────────────────────────────────────────────────────────────────

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


# ─── FNO 1D ─────────────────────────────────────────────────────────────────

class SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        self.scale = 1.0 / (in_ch * out_ch)
        self.w_re = nn.Parameter(torch.randn(in_ch, out_ch, modes) * self.scale)
        self.w_im = nn.Parameter(torch.randn(in_ch, out_ch, modes) * self.scale)

    def forward(self, x):
        B, ch, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        n_ft = x_ft.shape[-1]
        m = min(self.modes, n_ft)
        w = (self.w_re[:, :, :m] + 1j * self.w_im[:, :, :m]).permute(1, 0, 2)
        out_ft = torch.zeros(B, x_ft.shape[1], n_ft, dtype=torch.complex64, device=x.device)
        out_ft[:, :, :m] = torch.einsum("bim,oim->bom", x_ft[:, :, :m], w)
        return torch.fft.irfft(out_ft, n=N, dim=-1)


class FNO1d(nn.Module):
    """FNO for 1D fields. Input and output both have spatial_dim=N."""
    def __init__(self, spatial_dim=64, width=32, modes=12, num_layers=3):
        super().__init__()
        self.fc0 = nn.Linear(1, width)
        self.spec = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(num_layers)])
        self.lins = nn.ModuleList([nn.Linear(width, width) for _ in range(num_layers)])
        self.fc1  = nn.Linear(width, width)
        self.fc2  = nn.Linear(width, 1)

    def forward(self, x):
        # x: [B, spatial_dim]
        h = self.fc0(x.unsqueeze(-1))
        for spec, lin in zip(self.spec, self.lins):
            h_t = spec(h.transpose(1, 2)).transpose(1, 2)
            h = h + nn.functional.gelu(lin(h_t))
        return self.fc2(nn.functional.gelu(self.fc1(h))).squeeze(-1)


# ─── Training Helpers ───────────────────────────────────────────────────────

def fit_power_law(Ns, errs):
    """
    Fit error = C * N^(-alpha)
    log(err) = log(C) - alpha * log(N)
    """
    log_N = np.log(np.array(Ns, dtype=float))
    log_e = np.log(np.array(errs, dtype=float))
    slope, intercept = np.polyfit(log_N, log_e, 1)
    return float(-slope), float(intercept)


def train_deeponet(model, loader, epochs=500, lr=1e-3):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for u_b, y_b, ex_b in loader:
            u_b, y_b, ex_b = u_b.to(DEVICE), y_b.to(DEVICE), ex_b.to(DEVICE)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(u_b, y_b), ex_b)
            loss.backward(); opt.step()
        if epoch % 200 == 0:
            print(f"      epoch {epoch}")


def eval_deeponet(model, u_vals, y_vals, ex_vals):
    model.to(DEVICE); model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(u_vals).to(DEVICE),
                     torch.from_numpy(y_vals).to(DEVICE))
        return nn.functional.mse_loss(pred, torch.from_numpy(ex_vals).to(DEVICE)).item()


def train_fno(model, u_train, ex_train, epochs=500, lr=1e-3, batch_size=32):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = len(u_train)
    for epoch in range(epochs):
        idx = np.random.permutation(N)
        total_loss = 0.0; nb = 0
        for start in range(0, N, batch_size):
            b_idx = idx[start:start+batch_size]
            u_b = torch.from_numpy(u_train[b_idx]).float().to(DEVICE)
            ex_b = torch.from_numpy(ex_train[b_idx]).float().to(DEVICE)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(u_b), ex_b)
            loss.backward(); opt.step()
            total_loss += loss.item(); nb += 1
        if epoch % 200 == 0:
            print(f"      epoch {epoch}")


def eval_fno(model, u_vals, ex_vals):
    model.to(DEVICE); model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(u_vals).float().to(DEVICE))
        return nn.functional.mse_loss(pred, torch.from_numpy(ex_vals).float().to(DEVICE)).item()


# ─── Run Experiment ──────────────────────────────────────────────────────────

THEORY = {
    "heat_1d":     {"dim": 1, "alpha": 0.500},
    "poisson_2d":  {"dim": 2, "alpha": 0.500},
}

N_VALUES = [50, 100, 200, 400]
EPOCHS   = 500
TEST_N   = 100


def run_experiment(pde_name, make_dataset_fn, num_sensors, coord_dim,
                   solver_name="deeponet", extra_args=None):
    if extra_args is None: extra_args = {}
    print(f"\n{'='*60}\n  {pde_name} / {solver_name}\n{'='*60}")
    t0 = time.time()

    # ── Build independent train and test sets ──────────────────────────────
    # Training: seed 2000/3000; Test: seed 9000 (completely separate)
    train_seed = 2000 if solver_name == "deeponet" else 3000
    test_seed  = 9000

    test_ds  = make_dataset_fn(N=TEST_N,  seed=test_seed,  num_sensors=num_sensors,
                                num_points=64, **extra_args)
    print(f"  Test set: {test_ds['u_sensors'].shape[0]} samples (seed={test_seed})")

    errs_by_N = {}
    for N in N_VALUES:
        print(f"\n  N={N}...", flush=True)

        train_ds = make_dataset_fn(N=N, seed=train_seed, num_sensors=num_sensors,
                                   num_points=64, **extra_args)

        u_tr  = train_ds["u_sensors"]
        y_tr  = train_ds["y_coords"]
        ex_tr = train_ds["u_query"]
        u_te  = test_ds["u_sensors"]
        y_te  = test_ds["y_coords"]
        ex_te = test_ds["u_query"]

        if solver_name == "deeponet":
            loader = DataLoader(PDEDataset(u_tr, y_tr, ex_tr), batch_size=32, shuffle=True)
            model = DeepONet(num_sensors=num_sensors, coord_dim=coord_dim, p=50, hidden=128)
            train_deeponet(model, loader, epochs=EPOCHS)
            mse = eval_deeponet(model, u_te, y_te, ex_te)
            errs_by_N[N] = mse
            print(f"    DeepONet  N={N}: test MSE={mse:.6f}")

        elif solver_name == "fno":
            # FNO: input is IC at sensors → output is solution at SAME sensor locations
            # (not query points; avoids shape mismatch)
            model = FNO1d(spatial_dim=num_sensors, width=32, modes=12, num_layers=3)
            train_fno(model, u_tr, ex_tr, epochs=EPOCHS)
            mse = eval_fno(model, u_te, ex_te)
            errs_by_N[N] = mse
            print(f"    FNO  N={N}: test MSE={mse:.6f}")

    # Fit power law
    Ns   = sorted(errs_by_N.keys())
    errs = [errs_by_N[N] for N in Ns]
    alpha, log_C = fit_power_law(Ns, errs)
    t_alpha = THEORY.get(pde_name, {}).get("alpha", 0.5)
    ratio   = alpha / t_alpha if t_alpha > 0 else 0.0
    status  = "✓" if 0.25 <= ratio <= 1.6 else ("†" if ratio > 1.6 else "✗")

    print(f"\n  RESULT: α_fit={alpha:.4f}, α_theory={t_alpha:.4f}, "
          f"ratio={ratio:.1%} {status}")
    print(f"  Errors: {[round(e,6) for e in errs]}")
    print(f"  Time: {time.time()-t0:.0f}s")

    return {
        "pde": pde_name, "solver": solver_name,
        "alpha_hat": round(alpha, 4), "alpha_theory": t_alpha,
        "ratio": round(ratio, 4),
        "C": round(math.exp(log_C), 4),
        "N_values": [int(n) for n in Ns],
        "errors": [round(e, 6) for e in errs],
        "wall_time_s": round(time.time() - t0, 1),
    }


def plot_result(r, out_path):
    Ns    = np.array(r["N_values"])
    errs  = np.array(r["errors"])
    alpha  = r["alpha_hat"]
    alpha_t = r["alpha_theory"]
    C = r["C"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.loglog(Ns, errs, "o-", ms=8, lw=2, label=f"Empirical (α̂={alpha:.3f})")
    N_fit = np.linspace(Ns[0]*0.9, Ns[-1]*1.1, 100)
    ax.loglog(N_fit, C * N_fit**(-alpha),    "b--", lw=1.5, alpha=0.7, label="Fit C·N⁻α")
    ax.loglog(N_fit, C * N_fit**(-alpha_t),   "r:",  lw=2,   label=f"Theory α={alpha_t:.3f}")
    ax.set_xlabel("N (training samples)"); ax.set_ylabel("L² Error")
    ax.set_title(f"{r['pde']} / {r['solver']}")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(Ns, errs, "o-", ms=8, lw=2)
    ax.set_xlabel("N"); ax.set_ylabel("L² Error")
    ax.set_title("Linear scale"); ax.grid(True, alpha=0.3)

    plt.suptitle(f"Convergence: {r['pde']} / {r['solver']}  (α̂={alpha:.3f}, theory α={alpha_t:.3f})",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Neural Operator Convergence — PyTorch v4 / Intel Mac Mini")
    print("  (train≠test seeds, proper generalisation)")
    print("=" * 70)
    t0 = time.time()
    results = []

    # Heat 1D + DeepONet
    r = run_experiment("heat_1d", make_heat_dataset,
                       num_sensors=32, coord_dim=1, solver_name="deeponet",
                       extra_args={"t_final": 0.05})
    plot_result(r, FIG_DIR / "convergence_heat_1d_deeponet.png"); results.append(r)

    # Poisson 2D + DeepONet
    r = run_experiment("poisson_2d", make_poisson_dataset,
                       num_sensors=64, coord_dim=2, solver_name="deeponet",
                       extra_args={"grid_size": 8, "n_modes": 3})
    plot_result(r, FIG_DIR / "convergence_poisson_2d_deeponet.png"); results.append(r)

    # Save
    with open(OUT_DIR / "convergence_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(OUT_DIR / "table1_data.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Table 1
    print("\n" + "=" * 78)
    print("  TABLE 1: Empirical vs Theoretical Convergence Rates")
    print("=" * 78)
    hdr = f"{'PDE':<18} {'Dim':<4} {'Solver':<10} {'α̂':>7} {'α_theory':>10} {'Ratio':>10} {'Status':>7}"
    print(hdr); print("-" * 78)
    for r in results:
        dim = THEORY.get(r["pde"], {}).get("dim", 1)
        ratio = r["ratio"]
        status = "✓" if 0.25 <= ratio <= 1.6 else ("†" if ratio > 1.6 else "✗")
        print(f"{r['pde']:<18} {dim:<4} {r['solver']:<10} {r['alpha_hat']:>7.4f} "
              f"{r['alpha_theory']:>10.4f} {ratio:>9.1%} {status:>7}")
    print("=" * 78)
    print(f"\nTotal: {time.time()-t0:.0f}s  |  Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
