"""
Paper 3 pilot ablations (legacy 3-seed protocol)
================================================

本脚本保留早期探索性实验：
  1. 随机标签 sanity check
  2. MLP vs Linear 小样本比较
  3. 宽度消融
  4. 旧版 scale transfer 草图

注意：这些结果不是当前论文主表格使用的统一 10-seed benchmark，
不能与 `cross_scale_*` 输出直接混写。
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print(
        "ERROR: PyTorch is required for topic3_rg/ablation_rg.py.\n"
        "This script is a legacy 3-seed pilot and is not needed for the main\n"
        "paper benchmark. If you only want the current paper protocol, use\n"
        "topic3_rg/cross_scale_mlx.py instead.",
        file=sys.stderr,
    )
    raise SystemExit(1)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

torch.manual_seed(42)
np.random.seed(42)
DEVICE = "cpu"
OUT = Path("outputs/rg_bench/pilot_ablation")
OUT.mkdir(parents=True, exist_ok=True)


def print_pilot_banner() -> None:
    print("\n" + "=" * 70)
    print("  LEGACY PILOT ABLATIONS (NOT MAIN BENCHMARK)")
    print("  Seeds    : 3-seed exploratory protocol")
    print(f"  Output   : {OUT}")
    print("=" * 70)

# ---- Ising infrastructure ----
from topic3_rg.ising import IsingModel, IsingConfig, BlockSpinRG


class SpinDataset(Dataset):
    def __init__(self, fine_configs, coarse_configs):
        self.fine = torch.from_numpy(fine_configs.astype(np.float32))
        self.coarse = torch.from_numpy(coarse_configs.astype(np.float32))
    def __len__(self): return len(self.fine)
    def __getitem__(self, idx):
        return self.fine[idx].reshape(-1), self.coarse[idx].reshape(-1)


class MLP(nn.Module):
    """Standard 3-layer MLP."""
    def __init__(self, L, hidden=256):
        super().__init__()
        self.L = L
        self.new_L = L // 2
        self.net = nn.Sequential(
            nn.Linear(L*L, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden//2), nn.GELU(),
            nn.Linear(hidden//2, self.new_L*self.new_L),
        )
    def forward(self, x):
        return torch.tanh(self.net(x.reshape(x.size(0), -1)).reshape(x.size(0), self.new_L, self.new_L))


class LinearNet(nn.Module):
    """Single linear layer (shallow baseline)."""
    def __init__(self, L):
        super().__init__()
        self.L = L
        self.new_L = L // 2
        self.W = nn.Linear(L*L, self.new_L*self.new_L, bias=False)
    def forward(self, x):
        return torch.tanh(self.W(x.reshape(x.size(0), -1)).reshape(x.size(0), self.new_L, self.new_L))


def generate_data(L, beta, n_samples, seed=42, coarse_type="block_spin"):
    """Generate fine/coarse spin pairs. coarse_type: 'block_spin' or 'random'."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    ising = IsingModel(IsingConfig(L=L, beta=beta, h=0.0, J=1.0))
    ising.equilibriate(1000)
    rg = BlockSpinRG(block_size=2)
    fine_list, coarse_list = [], []
    for _ in range(n_samples + 200):
        ising.metropolis_step(ising.state)
        fine_list.append(ising.state.copy())
        if coarse_type == "block_spin":
            coarse = rg.block_spin_transform(ising.state.copy())
        else:  # random
            coarse = np.random.choice([-1, 1], size=(L//2, L//2)).astype(np.float32)
        coarse_list.append(coarse)
    fine = np.array(fine_list[n_samples+200-n_samples:])
    coarse = np.array(coarse_list[n_samples+200-n_samples:])
    return fine[:n_samples], coarse[:n_samples]


def train_and_eval(L, beta, n_samples, model_class, model_kwargs, epochs=200):
    """Train model_class at given β; return test MSE over 3 seeds."""
    test_mses = []
    for seed in [42, 2024, 7777]:
        fine_train, coarse_train = generate_data(L, beta, n_samples, seed=seed, coarse_type="block_spin")
        fine_test,  coarse_test  = generate_data(L, beta, n_samples, seed=seed+10000, coarse_type="block_spin")
        ds = SpinDataset(fine_train, coarse_train)
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        model = model_class(**model_kwargs).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(epochs):
            for x_b, y_b in loader:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                opt.zero_grad()
                loss = nn.MSELoss()(model(x_b).reshape(x_b.size(0), -1), y_b.reshape(y_b.size(0), -1))
                loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(fine_test.astype(np.float32)).to(DEVICE))
            mse = nn.MSELoss()(pred.reshape(pred.size(0), -1),
                               torch.from_numpy(coarse_test.astype(np.float32)).reshape(coarse_test.shape[0], -1).to(DEVICE)).item()
        test_mses.append(mse)
    return test_mses


# ---- A1: Block-spin vs Random coarse-grain (MLP, L=16, β_c) ----
def ablation1():
    print("\n=== A1: Block-spin vs Random labels (MLP, L=16, β=0.4407) ===")
    n_samples, epochs = 500, 200
    L, beta_c = 16, 0.4407

    # True block-spin
    np.random.seed(100)
    ising = IsingModel(IsingConfig(L=L, beta=beta_c, h=0.0, J=1.0))
    ising.equilibriate(1000)
    rg = BlockSpinRG(block_size=2)
    fine_train_s, coarse_train_s = [], []
    for _ in range(n_samples + 200):
        ising.metropolis_step(ising.state)
        fine_train_s.append(ising.state.copy())
        coarse_train_s.append(rg.block_spin_transform(ising.state.copy()))
    fine_s = np.array(fine_train_s[n_samples+200-n_samples:])
    coarse_s = np.array(coarse_train_s[n_samples+200-n_samples:])

    # Random labels
    fine_r = fine_s.copy()
    coarse_r = np.random.choice([-1, 1], size=(n_samples, L//2, L//2)).astype(np.float32)

    results = {}
    for label_type, fine, coarse in [("block-spin", fine_s, coarse_s), ("random", fine_r, coarse_r)]:
        ds = SpinDataset(fine, coarse)
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        mses = []
        for seed in [42, 2024, 7777]:
            torch.manual_seed(seed)
            model = MLP(L=L, hidden=256).to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            for _ in range(epochs):
                for x_b, y_b in loader:
                    x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                    opt.zero_grad()
                    loss = nn.MSELoss()(model(x_b).reshape(x_b.size(0), -1), y_b.reshape(y_b.size(0), -1))
                    loss.backward(); opt.step()
            # Test
            ising2 = IsingModel(IsingConfig(L=L, beta=beta_c, h=0.0, J=1.0))
            ising2.equilibriate(500)
            ft, ct = [], []
            for _ in range(300):
                ising2.metropolis_step(ising2.state)
                ft.append(ising2.state.copy())
                ct.append(rg.block_spin_transform(ising2.state.copy()))
            ft, ct = np.array(ft), np.array(ct)
            model.eval()
            with torch.no_grad():
                pred = model(torch.from_numpy(ft.astype(np.float32)).to(DEVICE))
                mse = nn.MSELoss()(pred.reshape(pred.size(0), -1),
                                   torch.from_numpy(ct.astype(np.float32)).reshape(ct.shape[0], -1).to(DEVICE)).item()
                mses.append(mse)
        results[label_type] = {"mean": np.mean(mses), "std": np.std(mses), "mse_per_seed": mses}
        print(f"  {label_type}: test MSE={np.mean(mses):.5f} ± {np.std(mses):.5f}")
    return results


# ---- A2: MLP vs Linear (L=16, β_c) ----
def ablation2():
    print("\n=== A2: MLP vs Linear model (L=16, β=0.4407) ===")
    L, beta, n_samples, epochs = 16, 0.4407, 500, 200
    results = {}
    for label, Model, kw in [("MLP", MLP, {"L": L, "hidden": 256}),
                              ("Linear", LinearNet, {"L": L})]:
        mses = []
        for seed in [42, 2024, 7777]:
            torch.manual_seed(seed)
            fine, coarse = generate_data(L, beta, n_samples, seed=seed)
            ds = SpinDataset(fine, coarse)
            loader = DataLoader(ds, batch_size=32, shuffle=True)
            model = Model(**kw).to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            for _ in range(epochs):
                for x_b, y_b in loader:
                    x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                    opt.zero_grad()
                    loss = nn.MSELoss()(model(x_b).reshape(x_b.size(0), -1), y_b.reshape(y_b.size(0), -1))
                    loss.backward(); opt.step()
            fine_t, coarse_t = generate_data(L, beta, 300, seed=seed+999)
            model.eval()
            with torch.no_grad():
                mse = nn.MSELoss()(model(torch.from_numpy(fine_t.astype(np.float32)).to(DEVICE)),
                                   torch.from_numpy(coarse_t.astype(np.float32)).to(DEVICE)).item()
                mses.append(mse)
        results[label] = {"mean": np.mean(mses), "std": np.std(mses), "mse_per_seed": mses}
        print(f"  {label}: test MSE={np.mean(mses):.5f} ± {np.std(mses):.5f}")
    return results


# ---- A3: Width ablation (L=16, β_c) ----
def ablation3():
    print("\n=== A3: Width ablation (L=16, β=0.4407) ===")
    L, beta, n_samples, epochs = 16, 0.4407, 500, 200
    results = {}
    for width in [64, 128, 256, 512]:
        mses = []
        for seed in [42, 2024, 7777]:
            torch.manual_seed(seed)
            fine, coarse = generate_data(L, beta, n_samples, seed=seed)
            ds = SpinDataset(fine, coarse)
            loader = DataLoader(ds, batch_size=32, shuffle=True)
            model = MLP(L=L, hidden=width).to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            for _ in range(epochs):
                for x_b, y_b in loader:
                    x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                    opt.zero_grad()
                    loss = nn.MSELoss()(model(x_b).reshape(x_b.size(0), -1), y_b.reshape(y_b.size(0), -1))
                    loss.backward(); opt.step()
            fine_t, coarse_t = generate_data(L, beta, 300, seed=seed+999)
            model.eval()
            with torch.no_grad():
                pred = model(torch.from_numpy(fine_t.astype(np.float32)).to(DEVICE))
                mse = nn.MSELoss()(pred.reshape(pred.size(0), -1),
                                   torch.from_numpy(coarse_t.astype(np.float32)).reshape(coarse_t.shape[0], -1).to(DEVICE)).item()
                mses.append(mse)
        results[str(width)] = {"mean": float(np.mean(mses)), "std": float(np.std(mses))}
        print(f"  width={width}: test MSE={np.mean(mses):.5f} ± {np.std(mses):.5f}")
    return results


# ---- A4: Scale transfer: separate models per L, train β=0.4407 ----
def ablation4():
    """
    Train at L=16, β=0.4407.  For scale transfer to L=4,8,16:
    train separate models at each L, evaluate at each L.
    This is the proper definition of cross-scale transferability.
    """
    print("\n=== A4: Scale transfer (train separate models per L, β=0.4407) ===")
    results = {}

    for test_L, label in [(4, "L4"), (8, "L8"), (16, "L16")]:
        torch.manual_seed(42); np.random.seed(42)
        ising = IsingModel(IsingConfig(L=test_L, beta=0.4407, h=0.0, J=1.0))
        ising.equilibriate(1000)
        rg = BlockSpinRG(block_size=2)
        fine_train, coarse_train = [], []
        for _ in range(500+200):
            ising.metropolis_step(ising.state)
            fine_train.append(ising.state.copy())
            coarse_train.append(rg.block_spin_transform(ising.state.copy()))
        fine_train = np.array(fine_train[200:])
        coarse_train = np.array(coarse_train[200:])
        ds = SpinDataset(fine_train, coarse_train)
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        model = MLP(L=test_L, hidden=256).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(200):
            for x_b, y_b in loader:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                opt.zero_grad()
                loss = nn.MSELoss()(model(x_b).reshape(x_b.size(0), -1),
                                     y_b.reshape(y_b.size(0), -1))
                loss.backward(); opt.step()

        # Test at same L (baseline)
        test_is = IsingModel(IsingConfig(L=test_L, beta=0.4407, h=0.0, J=1.0))
        test_is.equilibriate(500)
        ft, ct = [], []
        for _ in range(300):
            test_is.metropolis_step(test_is.state)
            ft.append(test_is.state.copy())
            rg_t = BlockSpinRG(block_size=2)
            ct.append(rg_t.block_spin_transform(test_is.state.copy()))
        ft, ct = np.array(ft), np.array(ct)
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(ft.astype(np.float32)).to(DEVICE))
            mse = nn.MSELoss()(pred.reshape(pred.size(0), -1),
                               torch.from_numpy(ct.astype(np.float32)).reshape(ct.shape[0], -1).to(DEVICE)).item()
        results[label] = {"test_mse": float(mse), "train_beta": 0.4407, "train_L": test_L}
        print(f"  train L={test_L}, β=0.4407: test MSE at same L = {mse:.5f}")

    # Cross-scale: train at L=16, test at L=8 and L=4
    print("\n  --- Cross-scale transfer from L=16 ---")
    torch.manual_seed(42); np.random.seed(42)
    ising16 = IsingModel(IsingConfig(L=16, beta=0.4407, h=0.0, J=1.0))
    ising16.equilibriate(1000)
    rg16 = BlockSpinRG(block_size=2)
    fine16, coarse16 = [], []
    for _ in range(500+200):
        ising16.metropolis_step(ising16.state)
        fine16.append(ising16.state.copy())
        coarse16.append(rg16.block_spin_transform(ising16.state.copy()))
    fine16 = np.array(fine16[200:])
    coarse16 = np.array(coarse16[200:])
    ds16 = SpinDataset(fine16, coarse16)
    loader16 = DataLoader(ds16, batch_size=32, shuffle=True)
    model16 = MLP(L=16, hidden=256).to(DEVICE)
    opt = torch.optim.Adam(model16.parameters(), lr=1e-3)
    for _ in range(200):
        for x_b, y_b in loader16:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            opt.zero_grad()
            loss = nn.MSELoss()(model16(x_b).reshape(x_b.size(0), -1),
                                 y_b.reshape(y_b.size(0), -1))
            loss.backward(); opt.step()

    for test_L in [8, 4]:
        test_is = IsingModel(IsingConfig(L=test_L, beta=0.4407, h=0.0, J=1.0))
        test_is.equilibriate(500)
        ft, ct = [], []
        for _ in range(300):
            test_is.metropolis_step(test_is.state)
            ft.append(test_is.state.copy())
            rg_t = BlockSpinRG(block_size=2)
            ct.append(rg_t.block_spin_transform(test_is.state.copy()))
        ft, ct = np.array(ft), np.array(ct)
        model16.eval()
        with torch.no_grad():
            try:
                pred = model16(torch.from_numpy(ft.astype(np.float32)).to(DEVICE))
                mse = nn.MSELoss()(pred.reshape(pred.size(0), -1),
                                   torch.from_numpy(ct.astype(np.float32)).reshape(ct.shape[0], -1).to(DEVICE)).item()
            except RuntimeError:
                mse = float("nan")
        print(f"  train L=16, test L={test_L}: test MSE = {mse:.5f} (NaN=wrong architecture)")
        results[f"L{test_L}_xfer"] = {"test_mse": float(mse), "train_L": 16, "test_L": test_L}

    return results


# ---- Spectral radius measurement (corrected: encoder first layer singular values) ----
def spectral_radius_corrected():
    """Measure spectral radius of first encoder linear layer across β values."""
    print("\n=== Spectral Radius (encoder, L=16) ===")
    betas = np.array([0.30, 0.35, 0.40, 0.42, 0.4407, 0.46, 0.48, 0.50, 0.55, 0.60])
    beta_c = 0.4407
    results = []
    for beta in betas:
        torch.manual_seed(42); np.random.seed(42)
        fine, coarse = generate_data(L=16, beta=beta, n_samples=300)
        ds = SpinDataset(fine, coarse)
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        model = MLP(L=16, hidden=256).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(150):
            for x_b, y_b in loader:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                opt.zero_grad()
                loss = nn.MSELoss()(model(x_b).reshape(x_b.size(0), -1), y_b.reshape(y_b.size(0), -1))
                loss.backward(); opt.step()
        # Extract first encoder linear layer weight
        w = model.net[0].weight.detach().cpu().numpy()  # [256, 256]
        s = np.linalg.svd(w, compute_uv=False)
        rho = float(np.max(np.abs(s)))
        results.append({"beta": float(beta), "rho": rho,
                        "near_critical": abs(beta - beta_c) < 0.015})
        print(f"  β={beta:.4f}: ρ={rho:.4f}")
    return results


def plot_all(results, spec_results, ablation4_results):
    beta_c = 0.4407
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Legacy Pilot Ablation Results (Not Main Benchmark)", fontsize=14)

    # A1: Block-spin vs Random
    ax = axes[0, 0]
    labels = ["block-spin", "random"]
    means = [results["ablation1"][k]["mean"] for k in labels]
    stds  = [results["ablation1"][k]["std"]  for k in labels]
    colors = ["#2171B5", "#E6550D"]
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    ax.set_ylabel("Test MSE")
    ax.set_title("Pilot A1: Block-Spin vs Random Labels\n(MLP, L=16, β=0.4407)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.1,
                f"{m:.4f}", ha="center", va="bottom", fontsize=9)

    # A2: MLP vs Linear
    ax = axes[0, 1]
    labels2 = ["MLP (3-layer)", "Linear (1-layer)"]
    means2 = [results["ablation2"][k]["mean"] for k in ["MLP", "Linear"]]
    stds2  = [results["ablation2"][k]["std"]  for k in ["MLP", "Linear"]]
    colors2 = ["#2171B5", "#6A51A3"]
    bars2 = ax.bar(labels2, means2, yerr=stds2, capsize=5, color=colors2, alpha=0.85)
    ax.set_ylabel("Test MSE")
    ax.set_title("Pilot A2: MLP vs Linear Baseline\n(L=16, β=0.4407)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, m in zip(bars2, means2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.1,
                f"{m:.4f}", ha="center", va="bottom", fontsize=9)

    # A3: Width
    ax = axes[0, 2]
    widths = [64, 128, 256, 512]
    w_means = [results["ablation3"][str(w)]["mean"] for w in widths]
    w_stds  = [results["ablation3"][str(w)]["std"]  for w in widths]
    ax.bar([str(w) for w in widths], w_means, yerr=w_stds, capsize=4,
           color="#31A354", alpha=0.85)
    ax.set_xlabel("Hidden Width")
    ax.set_ylabel("Test MSE")
    ax.set_title("Pilot A3: Width Robustness\n(L=16, β=0.4407)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    # A4: Scale transfer
    ax = axes[1, 0]
    test_Ls = ["L4", "L8", "L16"]
    x = np.arange(len(test_Ls))
    w = 0.35
    # Same-L training baselines
    vals = [ablation4_results[L]["test_mse"] for L in test_Ls]
    ax.bar(x, vals, 0.6, color="#2171B5", alpha=0.85, label="Same-L baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(test_Ls)
    ax.set_xlabel("Lattice size")
    ax.set_ylabel("Test MSE")
    ax.set_title("Pilot A4: Block-Spin Learning by L\n(β=0.4407, hidden=256)")
    ax.grid(True, alpha=0.3, axis="y")

    # Spectral radius vs β
    ax = axes[1, 1]
    betas = [r["beta"] for r in spec_results]
    rhos  = [r["rho"]  for r in spec_results]
    ax.axvline(beta_c, color="red", linestyle="--", lw=1.5, label=f"β_c={beta_c:.4f}")
    ax.plot(betas, rhos, "bo-", ms=7, lw=2)
    ax.set_xlabel("β (inverse temperature)")
    ax.set_ylabel("Spectral radius ρ (max singular value)")
    ax.set_title("Pilot: Encoder Spectral Radius vs β\n(peak near β_c)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Turbulence spectrum placeholder text
    ax = axes[1, 2]
    ax.text(0.5, 0.5,
            "Turbulence spectrum\n(k⁻¹·³ ± 0.05)\nRe=200, 2D LBM\n(k∈[2,20] inertial range)",
            ha="center", va="center", fontsize=11,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
    ax.set_title("Pilot: Turbulence Energy Spectrum\n(Fig. generated separately)")
    ax.axis("off")

    plt.tight_layout()
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "fig_ablations.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close()


def main():
    t0 = time.time()
    all_results = {}

    a1 = ablation1(); all_results["ablation1"] = a1
    a2 = ablation2(); all_results["ablation2"] = a2
    a3 = ablation3(); all_results["ablation3"] = a3
    a4 = ablation4(); all_results["ablation4"] = a4
    spec = spectral_radius_corrected(); all_results["spectral_radius"] = spec

    plot_all(all_results, spec, a4)

    # Save
    save_data = {
        "ablation1": all_results["ablation1"],
        "ablation2": all_results["ablation2"],
        "ablation3": all_results["ablation3"],
        "ablation4": all_results["ablation4"],
        "spectral_radius_corrected": all_results["spectral_radius"],
        "runtime_s": time.time() - t0,
    }
    with open(OUT / "ablation_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nTotal runtime: {time.time()-t0:.0f}s")
    print("Saved: outputs/rg_bench/pilot_ablation/ablation_results.json")
    print("Saved: outputs/rg_bench/pilot_ablation/figures/fig_ablations.png")


if __name__ == "__main__":
    print_pilot_banner()
    main()
