"""
FNO 1D Convergence Experiment — fixed shape mismatch.
======================================================
FNO requires input [B, N] and output [B, N] with the SAME spatial dim N.
We use: input = IC at 32 sensors, target = solution at the SAME 32 sensor
locations at t_final.  This is a well-defined operator mapping.
"""
from __future__ import annotations
import sys, json, time, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

OUT_DIR = Path("/Users/isaacliu/workspace/convergence-uq-rg/outputs/convergence")
FIG_DIR = OUT_DIR / "figures"
DEVICE  = "cpu"


def heat1d_exact(x, t, coeffs):
    val = np.zeros_like(x, dtype=np.float64)
    for k, a in enumerate(coeffs, 1):
        val += a * np.sin(k * math.pi * x) * math.exp(-(k**2) * math.pi**2 * t)
    return val.astype(np.float32)


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
    def __init__(self, spatial_dim=32, width=32, modes=12, num_layers=3):
        super().__init__()
        self.fc0 = nn.Linear(1, width)
        self.spec = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(num_layers)])
        self.lins = nn.ModuleList([nn.Linear(width, width) for _ in range(num_layers)])
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, 1)

    def forward(self, x):
        h = self.fc0(x.unsqueeze(-1))
        for spec, lin in zip(self.spec, self.lins):
            h_t = spec(h.transpose(1, 2)).transpose(1, 2)
            h = h + nn.functional.gelu(lin(h_t))
        return self.fc2(nn.functional.gelu(self.fc1(h))).squeeze(-1)


def fit_power_law(Ns, errs):
    log_N = np.log(np.array(Ns, dtype=float))
    log_e = np.log(np.array(errs, dtype=float))
    slope, intercept = np.polyfit(log_N, log_e, 1)
    return float(-slope), float(intercept)


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
            print(f"  epoch {epoch}: MSE={total_loss/nb:.6f}")


def main():
    print("=" * 60)
    print("  Heat 1D + FNO (shape-matched)")
    print("=" * 60)

    t0 = time.time()
    N_VALUES = [50, 100, 200, 400]
    EPOCHS = 500
    TEST_N = 100
    SPATIAL_DIM = 32   # same for input and output
    T_FINAL = 0.05

    # Fixed test set (seed 9000)
    np.random.seed(9000)
    sensor_x = np.linspace(0.05, 0.95, SPATIAL_DIM)
    test_u, test_ex = [], []
    for _ in range(TEST_N):
        c = np.random.randn(4)*0.4; c[0]=1.0
        ic = heat1d_exact(sensor_x, 0.0, c)
        uq = heat1d_exact(sensor_x, T_FINAL, c)
        test_u.append(ic); test_ex.append(uq)
    test_u = np.array(test_u, dtype=np.float32)
    test_ex = np.array(test_ex, dtype=np.float32)
    print(f"Test set: {test_u.shape} / {test_ex.shape}")

    errs_by_N = {}
    for N in N_VALUES:
        print(f"\n  N={N}...", flush=True)

        # Training set (seed 3000)
        np.random.seed(3000)
        train_u, train_ex = [], []
        for _ in range(N):
            c = np.random.randn(4)*0.4; c[0]=1.0
            ic = heat1d_exact(sensor_x, 0.0, c)
            uq = heat1d_exact(sensor_x, T_FINAL, c)
            train_u.append(ic); train_ex.append(uq)
        train_u = np.array(train_u, dtype=np.float32)
        train_ex = np.array(train_ex, dtype=np.float32)

        model = FNO1d(spatial_dim=SPATIAL_DIM, width=32, modes=12, num_layers=3)
        train_fno(model, train_u, train_ex, epochs=EPOCHS)

        model.eval()
        with torch.no_grad():
            mse = nn.functional.mse_loss(
                model(torch.from_numpy(test_u).float().to(DEVICE)),
                torch.from_numpy(test_ex).float().to(DEVICE)
            ).item()
        errs_by_N[N] = mse
        print(f"    N={N}: test MSE={mse:.6f}")

    # Fit
    Ns = sorted(errs_by_N.keys())
    errs = [errs_by_N[N] for N in Ns]
    alpha, log_C = fit_power_law(Ns, errs)
    ratio = alpha / 0.5

    print(f"\n  RESULT: α_fit={alpha:.4f}, α_theory=0.5000, ratio={ratio:.1%}")
    print(f"  Errors: {[round(e,6) for e in errs]}")

    # Save
    result = {
        "pde": "heat_1d", "solver": "fno",
        "alpha_hat": round(alpha, 4), "alpha_theory": 0.500,
        "ratio": round(ratio, 4),
        "C": round(math.exp(log_C), 4),
        "N_values": [int(n) for n in Ns],
        "errors": [round(e, 6) for e in errs],
        "wall_time_s": round(time.time() - t0, 1),
    }
    with open(OUT_DIR / "fno_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Plot
    Ns = np.array(result["N_values"])
    errs = np.array(result["errors"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    N_fit = np.linspace(Ns[0]*0.9, Ns[-1]*1.1, 100)
    C = result["C"]
    ax.loglog(Ns, errs, "o-", ms=8, lw=2, label=f"Empirical (α̂={alpha:.3f})")
    ax.loglog(N_fit, C * N_fit**(-alpha), "b--", lw=1.5, alpha=0.7, label="Fit")
    ax.loglog(N_fit, C * N_fit**(-0.5), "r:", lw=2, label="Theory α=0.5")
    ax.set_xlabel("N"); ax.set_ylabel("L² Error"); ax.set_title("Heat 1D / FNO")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    ax = axes[1]
    ax.plot(Ns, errs, "o-", ms=8, lw=2)
    ax.set_xlabel("N"); ax.set_ylabel("L² Error"); ax.set_title("Linear"); ax.grid(True, alpha=0.3)
    plt.suptitle(f"Heat 1D / FNO — α̂={alpha:.3f}, theory=0.500", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "convergence_heat_1d_fno.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure and results")
    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
