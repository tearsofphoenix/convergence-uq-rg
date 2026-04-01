"""
FNO 2D Convergence Experiment — Poisson equation on [0,1]²
===========================================================
Poisson 2D: u(x,y) = Σ a_mn sin((m+1)πx) sin((n+1)πy)
u_sensors: values at grid_size×grid_size locations → input [B, N]
u_query: values at num_points random (x,y) locations → output [B, num_points]

FNO 2D: spectral convolution in 2D.
Grid shape: [N, N] channels.
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


def poisson2d_exact(x, y, coeffs):
    n_m = coeffs.shape[0]
    val = np.zeros_like(x, dtype=np.float64)
    for m in range(n_m):
        for n in range(n_m):
            val += coeffs[m, n] * np.sin((m+1)*math.pi*x) * np.sin((n+1)*math.pi*y)
    return val.astype(np.float32)


# ─── FNO 2D ─────────────────────────────────────────────────────────────────

class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.scale = 1.0 / (in_ch * out_ch)
        self.w_re = nn.Parameter(torch.randn(in_ch, out_ch, modes1, modes2) * self.scale)
        self.w_im = nn.Parameter(torch.randn(in_ch, out_ch, modes1, modes2) * self.scale)

    def forward(self, x):
        # x: [B, ch, H, W]
        B, ch, H, W = x.shape
        x_ft = torch.fft.fft2(x, dim=(2, 3))
        m1, m2 = min(self.modes1, H//2), min(self.modes2, W//2)
        # w: [in_ch, out_ch, m1, m2]
        w_complex = (self.w_re[:, :, :m1, :m2] + 1j * self.w_im[:, :, :m1, :m2])
        # Extract low-freq modes: [B, in_ch, m1, m2]
        x_m = x_ft[:, :, :m1, :m2]
        # Multiply and sum over in_ch: [B, out_ch, m1, m2]
        # x_m: [B, i, w, h], w_complex: [i, o, w, h]
        # Contract over i, keep b,o,w,h
        out_m = torch.einsum("biwh,iowh->bowh", x_m, w_complex)
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :m1, :m2] = out_m
        out_ft[:, :, -m1:, -m2:] = out_m.conj()
        return torch.fft.irfft2(out_ft, s=(H, W), dim=(2, 3))


class FNO2d(nn.Module):
    def __init__(self, width=20, modes1=8, modes2=8, num_layers=3):
        super().__init__()
        self.fc0 = nn.Linear(1, width)
        self.specs = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2)
                                     for _ in range(num_layers)])
        self.lins = nn.ModuleList([nn.Linear(width, width) for _ in range(num_layers)])
        self.fc1  = nn.Linear(width, width)
        self.fc2  = nn.Linear(width, 1)
        self.width = width

    def forward(self, x):
        # x: [B, H, W]
        h = self.fc0(x.unsqueeze(-1))           # [B, H, W, 1] → [B, H, W, width]
        h = h.permute(0, 3, 1, 2)               # [B, width, H, W]
        for spec, lin in zip(self.specs, self.lins):
            h_t = spec(h)                          # [B, width, H, W]
            B2, C2, H2, W2 = h_t.shape
            h_t_flat = h_t.permute(0, 2, 3, 1).reshape(B2 * H2 * W2, C2)  # [B*H*W, width]
            h_t_out  = lin(h_t_flat).reshape(B2, H2, W2, C2).permute(0, 3, 1, 2)  # [B, width, H, W]
            h = h + nn.functional.gelu(h_t_out)
        h = h.permute(0, 2, 3, 1)               # [B, H, W, width]
        return self.fc2(nn.functional.gelu(self.fc1(h))).squeeze(-1)  # [B, H, W]


def fit_power_law(Ns, errs):
    log_N = np.log(np.array(Ns, dtype=float))
    log_e = np.log(np.array(errs, dtype=float))
    slope, intercept = np.polyfit(log_N, log_e, 1)
    return float(-slope), float(intercept)


def make_dataset(N, grid_size=8, n_modes=3, seed=42):
    """Return (u_sensors [N, grid_size, grid_size], u_query [N, num_points])"""
    np.random.seed(seed)
    xg = np.linspace(0.1, 0.9, grid_size)
    yg = np.linspace(0.1, 0.9, grid_size)
    SX, SY = np.meshgrid(xg, yg, indexing='ij')

    u_sensors_l, u_query_l = [], []
    for i in range(N):
        coeffs = np.random.randn(n_modes, n_modes) * 0.2
        coeffs[0, 0] = 1.0
        u_full = poisson2d_exact(SX, SY, coeffs)
        # sample query points on the SAME grid
        uq = u_full.flatten()   # for FNO: same grid_size×grid_size output
        u_sensors_l.append(u_full.astype(np.float32))
        u_query_l.append(u_full.astype(np.float32))   # same grid for FNO

    return {
        "u_sensors": np.array(u_sensors_l, dtype=np.float32),
        "u_query":   np.array(u_query_l,   dtype=np.float32),
    }


def train_fno2d(model, u_train, ex_train, epochs=500, lr=1e-3, batch_size=16):
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
            pred = model(u_b)           # [B, H, W]
            loss = nn.functional.mse_loss(pred, ex_b)
            loss.backward(); opt.step()
            total_loss += loss.item(); nb += 1
        if epoch % 200 == 0:
            print(f"  epoch {epoch}: MSE={total_loss/nb:.6f}")


def eval_fno2d(model, u_vals, ex_vals):
    model.to(DEVICE); model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(u_vals).float().to(DEVICE))
        return nn.functional.mse_loss(pred, torch.from_numpy(ex_vals).float().to(DEVICE)).item()


def main():
    print("=" * 60)
    print("  Poisson 2D + FNO")
    print("=" * 60)
    t0 = time.time()
    GRID_SIZE = 8
    N_VALUES  = [50, 100, 200, 400]
    EPOCHS    = 500
    TEST_N    = 100
    TRAIN_SEED = 3000
    TEST_SEED  = 9000

    # Fixed test set
    test_ds = make_dataset(TEST_N, grid_size=GRID_SIZE, n_modes=3, seed=TEST_SEED)
    print(f"Test set: {test_ds['u_sensors'].shape} / {test_ds['u_query'].shape}")
    test_u  = test_ds["u_sensors"]
    test_ex = test_ds["u_query"]

    errs_by_N = {}
    for N in N_VALUES:
        print(f"\n  N={N}...", flush=True)
        train_ds = make_dataset(N, grid_size=GRID_SIZE, n_modes=3, seed=TRAIN_SEED)
        train_u  = train_ds["u_sensors"]
        train_ex = train_ds["u_query"]

        model = FNO2d(width=20, modes1=6, modes2=6, num_layers=3)
        train_fno2d(model, train_u, train_ex, epochs=EPOCHS)
        mse = eval_fno2d(model, test_u, test_ex)
        errs_by_N[N] = mse
        print(f"    N={N}: test MSE={mse:.6f}")

    Ns   = sorted(errs_by_N.keys())
    errs = [errs_by_N[N] for N in Ns]
    alpha, log_C = fit_power_law(Ns, errs)
    ratio = alpha / 0.5

    print(f"\n  RESULT: α_fit={alpha:.4f}, α_theory=0.500, ratio={ratio:.1%}")
    print(f"  Errors: {[round(e,6) for e in errs]}")

    result = {
        "pde": "poisson_2d", "solver": "fno",
        "alpha_hat": round(alpha, 4), "alpha_theory": 0.500,
        "ratio": round(ratio, 4),
        "C": round(math.exp(log_C), 4),
        "N_values": [int(n) for n in Ns],
        "errors": [round(e, 6) for e in errs],
        "wall_time_s": round(time.time() - t0, 1),
    }
    with open(OUT_DIR / "fno_poisson_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Plot
    Ns_arr = np.array(result["N_values"])
    errs_arr = np.array(result["errors"])
    C_val = result["C"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    N_fit = np.linspace(Ns_arr[0]*0.9, Ns_arr[-1]*1.1, 100)
    ax.loglog(Ns_arr, errs_arr, "o-", ms=8, lw=2, label=f"Empirical (α̂={alpha:.3f})")
    ax.loglog(N_fit, C_val * N_fit**(-alpha), "b--", lw=1.5, alpha=0.7, label="Fit")
    ax.loglog(N_fit, C_val * N_fit**(-0.5), "r:", lw=2, label="Theory α=0.5")
    ax.set_xlabel("N"); ax.set_ylabel("L² Error"); ax.set_title("Poisson 2D / FNO")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    ax = axes[1]
    ax.plot(Ns_arr, errs_arr, "o-", ms=8, lw=2)
    ax.set_xlabel("N"); ax.set_ylabel("L² Error"); ax.set_title("Linear"); ax.grid(True, alpha=0.3)
    plt.suptitle(f"Poisson 2D / FNO — α̂={alpha:.3f}, theory=0.500", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "convergence_poisson_2d_fno.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure")
    print(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
