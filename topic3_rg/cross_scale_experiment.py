"""
Cross-Scale Transfer Experiment (R2) for Paper 3 RG × NN
=========================================================
Definitive R2 experiment: can a model trained at L=8 generalize to L=16?

Design:
  Part A — Same-L baselines (all L ∈ {4,8,16}, β ∈ {0.30, βc, 0.60}):
    MLP, Linear, CNN, RGMLP × 10 seeds → 3×3×4×10 = 360 runs

  Part B — Cross-scale transfer (the core R2 test):
    Train at L=8, test at L=16 (both 256→64 architecture).
    L=8 configs are zero-padded to 256-dim to match L=16 input dimension.
    Tests: does training at smaller scale help or hurt generalization?
    If RG-like: cross-scale error should be comparable to within-L=16.
    If not RG-like: cross-scale error should be substantially worse.

  Part C — RG Equivariance + Temperature Dependence (n=5 seeds)

Outputs:
  cross_scale_raw.csv        — all individual run results
  cross_scale_statistics.json — statistical tests (Welch t, Mann-Whitney, permutation, CI)
  figures/                   — heatmaps, scale-distance plots
"""
from __future__ import annotations
import sys, os
# Force line-buffered stdout even under nohup/background
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
import sys, os, json, time, itertools
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from topic3_rg.ising import IsingModel, IsingConfig, BlockSpinRG

OUT_DIR = Path("/Users/isaacliu/workspace/convergence-uq-rg/outputs/rg_bench/cross_scale")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─── Models ──────────────────────────────────────────────────────────────────

class FlatMLP(nn.Module):
    """MLP with fixed input dim (256) and output dim (64)."""
    def __init__(self, L_in: int = 256, L_out: int = 64, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(L_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, L_out),
        )
    def forward(self, x):
        # x: [B, 256]
        return torch.tanh(self.net(x))


class LinearModel(nn.Module):
    """Linear model with fixed dimensions."""
    def __init__(self, L_in: int = 256, L_out: int = 64):
        super().__init__()
        self.linear = nn.Linear(L_in, L_out)
    def forward(self, x):
        return torch.tanh(self.linear(x))


class CNNBlockSpin(nn.Module):
    """CNN: L×L → L/2×L/2 block-spin. Works for any power-of-2 L ≥ 4."""
    def __init__(self, L: int = 16):
        super().__init__()
        self.L = L
        new_L = L // 2
        # Channel width scales with L to keep params reasonable
        ch = max(8, L * 2)
        self.block_conv = nn.Conv2d(1, ch, kernel_size=2, stride=2)  # L→L/2
        self.net = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(ch, ch * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch * 2, 1, kernel_size=1),
        )
    def forward(self, x):
        # x can be [B, L*L] (flattened) or [B, L, L] (2D)
        if x.dim() == 2:
            L = int(np.sqrt(x.shape[1]))
            x = x.reshape(x.shape[0], L, L)
        h = x.unsqueeze(1)                          # [B,1,L,L]
        h = self.block_conv(h)                      # [B,ch,L/2,L/2]
        h = self.net(h)                             # [B,1,L/2,L/2]
        out = h.squeeze(1)                          # [B,L/2,L/2]
        return torch.tanh(out.reshape(x.shape[0], -1))  # [B, (L/2)^2]


class RGInformedMLP(nn.Module):
    """MLP with block-average initialization for any L (default 16)."""
    def __init__(self, L: int = 16, hidden: int = 256):
        super().__init__()
        self.L = L
        new_L = L // 2
        w_init = torch.zeros(hidden, L * L)
        for h in range(hidden):
            for bi in range(new_L):
                for bj in range(new_L):
                    k = (bi * new_L + bj) % hidden
                    for di in range(2):
                        for dj in range(2):
                            gi = (bi * 2 + di) % L
                            gj = (bj * 2 + dj) % L
                            w_init[h, gi * L + gj] = 0.25
        self.encoder = nn.Linear(L * L, hidden)
        with torch.no_grad():
            self.encoder.weight.copy_(w_init * 4.0)
            self.encoder.bias.zero_()
        self.rg_transform = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(nn.Linear(hidden // 2, new_L * new_L))
    def forward(self, x):
        # x: [B, L*L]
        if x.dim() == 3:
            batch = x.shape[0]
            x = x.reshape(batch, -1)
        else:
            batch = x.shape[0]
        h = self.rg_transform(self.encoder(x))
        return torch.tanh(self.decoder(h))


# ─── Dataset ─────────────────────────────────────────────────────────────────

class SpinDataset(Dataset):
    def __init__(self, fine_configs, coarse_configs):
        self.fine = torch.from_numpy(fine_configs.astype(np.float32))
        self.coarse = torch.from_numpy(coarse_configs.astype(np.float32))
    def __len__(self):
        return len(self.fine)
    def __getitem__(self, idx):
        return self.fine[idx], self.coarse[idx]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def zero_pad_to_256(fine_configs: np.ndarray) -> np.ndarray:
    """
    Zero-pad fine_configs from 64-dim (L=8) to 256-dim (L=16).
    The coarse labels stay at 16-dim (L=8 → 4, or L=16 → 8 depending on context).
    For cross-scale: we use L=16 target, so coarse is always 64-dim.
    """
    n = len(fine_configs)
    padded = np.zeros((n, 256), dtype=np.float32)
    padded[:, :64] = fine_configs  # L=8 configs fit in first 64 positions
    return padded


# ─── Training ────────────────────────────────────────────────────────────────

def train_and_eval(model_cls, beta, n_train, n_test, epochs, batch_size,
                   seed, device, L_data, L_target, model_L_in, model_L_out):
    """
    Train model_cls at given beta.
    - L_data: lattice size for data generation
    - L_target: target lattice size (coarse-grained output dimension)
    - model_L_in/model_L_out: model input/output dimensions
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    rg = BlockSpinRG(block_size=2)

    # Generate training data
    # For training: fine configs at L_data, coarse labels at L_target
    # (L_target = the scale the model is trained to output)
    ising = IsingModel(IsingConfig(L=L_data, beta=beta, h=0.0, J=1.0))
    ising.equilibriate(1000)
    fine_train, coarse_train = [], []

    # Cross-scale: also need configs at L_target to compute coarse labels
    if L_data != L_target:
        ising_target = IsingModel(IsingConfig(L=L_target, beta=beta, h=0.0, J=1.0))
        ising_target.equilibriate(1000)
        target_configs = []
        for _ in range(n_train + 200):
            ising_target.metropolis_step(ising_target.state)
            target_configs.append(ising_target.state.copy())
        target_configs = np.array(target_configs[:n_train + 200])

    for i in range(n_train + 200):
        ising.metropolis_step(ising.state)
        fine = ising.state.copy()
        fine_train.append(fine.flatten())
        if L_data == L_target:
            coarse = rg.block_spin_transform(fine.copy())
        else:
            # For cross-scale: coarsegraining of a corresponding L_target config
            coarse = rg.block_spin_transform(target_configs[i].copy())
        coarse_train.append(coarse.flatten())
    fine_train = np.array(fine_train[:n_train])
    coarse_train = np.array(coarse_train[:n_train])

    # Generate test data (always at L_target scale for evaluation)
    ising_test = IsingModel(IsingConfig(L=L_target, beta=beta, h=0.0, J=1.0))
    ising_test.equilibriate(1000)
    fine_test, coarse_test = [], []
    for _ in range(n_test + 100):
        ising_test.metropolis_step(ising_test.state)
        fine = ising_test.state.copy()
        # BUG FIX: coarsegraining is always at L_target scale (the evaluation scale),
        # not L_data. For cross-scale (L_data=8, L_target=16): the model outputs
        # 64-dim (coarse at L=16), so test labels must also be coarse at L=16.
        coarse = rg.block_spin_transform(ising_test.state.copy())
        fine_test.append(fine.flatten())
        coarse_test.append(coarse.flatten())
    fine_test = np.array(fine_test[:n_test])
    coarse_test = np.array(coarse_test[:n_test])

    # Instantiate model
    if model_cls == "MLP":
        model = FlatMLP(L_in=model_L_in, L_out=model_L_out)
    elif model_cls == "Linear":
        model = LinearModel(L_in=model_L_in, L_out=model_L_out)
    elif model_cls == "CNN":
        # L = sqrt(model_L_in)
        L_model = int(np.sqrt(model_L_in))
        model = CNNBlockSpin(L=L_model)
    elif model_cls == "RGMLP":
        L_model = int(np.sqrt(model_L_in))
        model = RGInformedMLP(L=L_model)
    else:
        raise ValueError(model_cls)
    model = model.to(device)

    # Prepare training data
    if model_L_in == 256 and L_data == 8:
        # Cross-scale: zero-pad L=8 configs to 256-dim
        fine_train_padded = zero_pad_to_256(fine_train)
    else:
        fine_train_padded = fine_train

    train_ds = SpinDataset(fine_train_padded, coarse_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(x_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()

    # Evaluate on L_target test data
    model.eval()
    with torch.no_grad():
        # Test data is always L_target (native scale)
        fine_t = torch.from_numpy(fine_test.astype(np.float32)).to(device)
        coarse_t = torch.from_numpy(coarse_test.astype(np.float32)).to(device)
        pred_test = model(fine_t)
        test_mse = criterion(pred_test, coarse_t).item()

        # Train MSE (on padded training data)
        fine_t_train = torch.from_numpy(fine_train_padded.astype(np.float32)).to(device)
        coarse_t_train = torch.from_numpy(coarse_train.astype(np.float32)).to(device)
        pred_train = model(fine_t_train)
        train_mse = criterion(pred_train, coarse_t_train).item()

    return float(train_mse), float(test_mse)


# ─── Statistical tests ──────────────────────────────────────────────────────

def welch_ttest(g1, g2):
    t, p = stats.ttest_ind(g1, g2, equal_var=False)
    return float(t), float(p)

def mann_whitney_u(g1, g2):
    stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
    return float(stat), float(p)

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    s1, s2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
    return float((np.mean(g1) - np.mean(g2)) / pooled) if pooled > 1e-10 else 0.0

def permutation_test(g1, g2, n_perm=10000):
    obs = np.mean(g1) - np.mean(g2)
    combined = np.concatenate([g1, g2])
    n1 = len(g1)
    cnt = 0
    for _ in range(n_perm):
        np.random.shuffle(combined)
        if np.mean(combined[:n1]) - np.mean(combined[n1:]) <= obs:
            cnt += 1
    return cnt / n_perm, float(obs)

def bootstrap_ci(data, stat=np.mean, n_boot=10000, ci=0.95):
    vals = [stat(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    a = (1 - ci) / 2
    return float(np.percentile(vals, a*100)), float(np.percentile(vals, (1-a)*100))


# ─── Main experiment ──────────────────────────────────────────────────────────

def run_cross_scale_experiment(n_seeds=10, n_train=500, n_test=300,
                                epochs=200, batch_size=32):
    print("\n" + "="*70)
    print("  CROSS-SCALE TRANSFER EXPERIMENT (R2)")
    print(f"  {n_seeds} seeds, N_train={n_train}, epochs={epochs}")
    print("="*70)

    device = "cpu"
    betas = [0.30, 0.4407, 0.60]
    beta_labels = {0.30: "β=0.30 (disordered)", 0.4407: "β_c=0.4407 (critical)", 0.60: "β=0.60 (ordered)"}
    L_values = [4, 8, 16]

    seeds = list(range(42, 42 + n_seeds))
    results = []

    # ══ Part A: Same-L experiments ══════════════════════════════════════════
    print("\n[Part A] Same-L within-scale baselines")
    print("  Models: MLP, Linear, CNN, RGMLP")
    print("  L ∈ {4,8,16}, β ∈ {0.30, βc, 0.60}, 10 seeds each")

    part_a_configs = []
    for L in L_values:
        for beta in betas:
            for model in ["MLP", "Linear", "CNN", "RGMLP"]:
                for seed in seeds:
                    part_a_configs.append({
                        "L_data": L, "L_target": L,
                        "model_L_in": L*L, "model_L_out": (L//2)*(L//2),
                        "beta": beta, "model": model, "seed": seed,
                    })

    t0 = time.time()
    total = len(part_a_configs)
    for i, cfg in enumerate(part_a_configs):
        if (i+1) % 40 == 1:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"\n  A progress: {i+1}/{total} ({100*(i+1)/total:.1f}%) "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

        train_mse, test_mse = train_and_eval(
            model_cls=cfg["model"], beta=cfg["beta"],
            n_train=n_train, n_test=n_test,
            epochs=epochs, batch_size=batch_size,
            seed=cfg["seed"], device=device,
            L_data=cfg["L_data"], L_target=cfg["L_target"],
            model_L_in=cfg["model_L_in"], model_L_out=cfg["model_L_out"],
        )
        results.append({**cfg, "train_mse": train_mse, "test_mse": test_mse,
                        "scale_distance": 0, "same_L": True, "part": "A"})

    # ══ Part B: Cross-scale experiment ════════════════════════════════════════
    print("\n\n[Part B] Cross-scale: L=8→L=16 transfer")
    print("  Train at L=8 (zero-padded to 256-dim), test at L=16")
    print("  Models: MLP, Linear  (CNN/RGMLP require fixed architecture)")

    part_b_configs = []
    for beta in betas:
        for model in ["MLP", "Linear"]:
            for seed in seeds:
                part_b_configs.append({
                    "L_data": 8, "L_target": 16,
                    "model_L_in": 256, "model_L_out": 64,
                    "beta": beta, "model": model, "seed": seed,
                })

    total_b = len(part_b_configs)
    for i, cfg in enumerate(part_b_configs):
        if (i+1) % 20 == 1:
            elapsed = time.time() - t0
            rate = (total + i + 1) / elapsed if elapsed > 0 else 0
            eta = (total + total_b - i - 1) / rate if rate > 0 else 0
            print(f"\n  B progress: {i+1}/{total_b} "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

        train_mse, test_mse = train_and_eval(
            model_cls=cfg["model"], beta=cfg["beta"],
            n_train=n_train, n_test=n_test,
            epochs=epochs, batch_size=batch_size,
            seed=cfg["seed"], device=device,
            L_data=cfg["L_data"], L_target=cfg["L_target"],
            model_L_in=cfg["model_L_in"], model_L_out=cfg["model_L_out"],
        )
        results.append({**cfg, "train_mse": train_mse, "test_mse": test_mse,
                        "scale_distance": 8, "same_L": False, "part": "B"})

    elapsed_total = time.time() - t0
    print(f"\n\n  Total: {len(results, flush=True)} runs in {elapsed_total:.0f}s "
          f"({elapsed_total/len(results):.1f}s/run, "
          f"{elapsed_total/3600:.2f}h)")

    # ── Save raw results ───────────────────────────────────────────────────
    import csv
    df_path = OUT_DIR / "cross_scale_raw.csv"
    with open(df_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved: {df_path}", flush=True)

    # ── Statistical analysis ─────────────────────────────────────────────────
    print("\n  === STATISTICAL TESTS ===", flush=True)
    stat_results = {}

    # Same-L: at each (L, beta), compare MLP vs Linear
    import pandas as pd
    df = pd.DataFrame(results)
    df_same = df[df["same_L"]].copy()

    for L in L_values:
        for beta in betas:
            sub = df_same[(df_same["L_data"] == L) & (df_same["beta"] == beta)]
            for model in ["MLP", "Linear", "CNN", "RGMLP"]:
                grp = sub[sub["model"] == model]["test_mse"]
                if len(grp) > 0:
                    key = f"L={L}_beta={beta:.4f}_{model}"
                    lo, hi = bootstrap_ci(grp.values)
                    stat_results[key] = {
                        "L": L, "beta": beta, "model": model,
                        "mean": float(grp.mean()), "std": float(grp.std(ddof=1)),
                        "n": len(grp), "ci_95": (float(lo), float(hi)),
                    }

            # MLP vs Linear comparison
            mlp_scores = sub[sub["model"] == "MLP"]["test_mse"].values
            lin_scores = sub[sub["model"] == "Linear"]["test_mse"].values
            if len(mlp_scores) >= 2 and len(lin_scores) >= 2:
                t_stat, t_pval = welch_ttest(mlp_scores, lin_scores)
                u_stat, u_pval = mann_whitney_u(mlp_scores, lin_scores)
                perm_p, obs_diff = permutation_test(mlp_scores, lin_scores)
                d = cohens_d(lin_scores, mlp_scores)
                key = f"L={L}_beta={beta:.4f}_MLPvsLinear"
                stat_results[key] = {
                    "MLP_mean": float(np.mean(mlp_scores)),
                    "MLP_std": float(np.std(mlp_scores, ddof=1)),
                    "Linear_mean": float(np.mean(lin_scores)),
                    "Linear_std": float(np.std(lin_scores, ddof=1)),
                    "Welch_t": t_stat, "Welch_p": t_pval,
                    "MannWhitney_U": u_stat, "MannWhitney_p": u_pval,
                    "Permutation_p": perm_p,
                    "Cohens_d": d,
                    "obs_diff": obs_diff,
                }
                print(f"\n  L={L}, β={beta:.4f}:", flush=True)
                print(f"    MLP   : {np.mean(mlp_scores, flush=True):.4f} ± {np.std(mlp_scores,ddof=1):.4f}")
                print(f"    Linear: {np.mean(lin_scores, flush=True):.4f} ± {np.std(lin_scores,ddof=1):.4f}")
                print(f"    Welch t={t_stat:.3f}, p={t_pval:.4f}", flush=True)
                print(f"    Mann-Wh U={u_stat:.1f}, p={u_pval:.4f}", flush=True)
                print(f"    Permutation p={perm_p:.4f}, Cohen d={d:.3f}", flush=True)

    # Cross-scale comparison
    print("\n  Cross-scale (L=8→L=16, flush=True) vs Same-L=16:")
    df_cross = df[df["part"] == "B"].copy()
    df_L16_same = df_same[(df_same["L_data"] == 16) & (df_same["beta"] == beta)]
    for model in ["MLP", "Linear"]:
        for beta in betas:
            cross_scores = df_cross[(df_cross["model"] == model) &
                                    (df_cross["beta"] == beta)]["test_mse"].values
            same_scores = df_same[(df_same["model"] == model) &
                                  (df_same["L_data"] == 16) &
                                  (df_same["beta"] == beta)]["test_mse"].values
            if len(cross_scores) >= 2 and len(same_scores) >= 2:
                key = f"CrossScale_{model}_beta={beta:.4f}"
                stat_results[key] = {
                    "cross_scale_mean": float(np.mean(cross_scores)),
                    "cross_scale_std": float(np.std(cross_scores, ddof=1)),
                    "same_L16_mean": float(np.mean(same_scores)),
                    "same_L16_std": float(np.std(same_scores, ddof=1)),
                    "degradation": float(np.mean(cross_scores) / (np.mean(same_scores) + 1e-10)),
                }
                print(f"  {model} β={beta:.4f}: "
                      f"cross-scale={np.mean(cross_scores, flush=True):.4f}±{np.std(cross_scores,ddof=1):.4f} "
                      f"vs same-L16={np.mean(same_scores):.4f}±{np.std(same_scores,ddof=1):.4f} "
                      f"ratio={np.mean(cross_scores)/(np.mean(same_scores)+1e-10):.2f}x")

    stats_path = OUT_DIR / "cross_scale_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stat_results, f, indent=2, default=str)
    print(f"\n  Saved: {stats_path}", flush=True)

    # ── Figures ─────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Figure 1: Same-L bar chart at β_c
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for col, beta in enumerate(betas):
            ax = axes[col]
            models = ["MLP", "Linear", "CNN", "RGMLP"]
            means = []
            stds = []
            for model in models:
                grp = df_same[(df_same["L_data"] == 16) & (df_same["beta"] == beta) &
                              (df_same["model"] == model)]["test_mse"]
                means.append(grp.mean() if len(grp) > 0 else 0)
                stds.append(grp.std(ddof=1) if len(grp) > 0 else 0)
            bars = ax.bar(models, means, yerr=stds, capsize=5,
                          color=["steelblue","coral","forestgreen","gold"], alpha=0.8)
            ax.set_ylabel("Test MSE")
            ax.set_title(beta_labels[beta])
            ax.set_ylim(0, max(max(means)+max(stds)*2, 2.0))
            ax.grid(True, alpha=0.3, axis="y")
            for bar, m, s in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.02,
                        f"{m:.3f}", ha="center", va="bottom", fontsize=8)
        plt.suptitle("Same-L=16 Test MSE by Model and Temperature (10 seeds)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "model_comparison_bars.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {FIG_DIR}/model_comparison_bars.png", flush=True)

        # Figure 2: Cross-scale degradation
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for col, beta in enumerate(betas):
            ax = axes[col]
            labels, cross_means, cross_stds, same_means, same_stds = [], [], [], [], []
            for model in ["MLP", "Linear"]:
                cs = df_cross[(df_cross["model"] == model) &
                              (df_cross["beta"] == beta)]["test_mse"].values
                ss = df_same[(df_same["model"] == model) &
                             (df_same["L_data"] == 16) &
                             (df_same["beta"] == beta)]["test_mse"].values
                labels.append(model)
                cross_means.append(np.mean(cs) if len(cs) else 0)
                cross_stds.append(np.std(cs, ddof=1) if len(cs) else 0)
                same_means.append(np.mean(ss) if len(ss) else 0)
                same_stds.append(np.std(ss, ddof=1) if len(ss) else 0)
            x = np.arange(len(labels))
            w = 0.35
            ax.bar(x - w/2, same_means, w, yerr=same_stds, capsize=4,
                   label="Same-L=16", color="steelblue", alpha=0.8)
            ax.bar(x + w/2, cross_means, w, yerr=cross_stds, capsize=4,
                   label="Cross L=8→16", color="coral", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel("Test MSE")
            ax.set_title(beta_labels[beta])
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
        plt.suptitle("Cross-Scale Transfer Degradation: L=8→16 vs Same-L=16 (10 seeds)",
                     fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "cross_scale_degradation.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {FIG_DIR}/cross_scale_degradation.png", flush=True)

        # Figure 3: All same-L results heatmap
        for model in ["MLP", "Linear"]:
            fig, ax = plt.subplots(figsize=(8, 6))
            matrix = np.full((len(L_values), len(betas)), np.nan)
            for i, L in enumerate(L_values):
                for j, beta in enumerate(betas):
                    grp = df_same[(df_same["L_data"] == L) &
                                  (df_same["beta"] == beta) &
                                  (df_same["model"] == model)]["test_mse"]
                    if len(grp) > 0:
                        matrix[i, j] = grp.mean()
            im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
            ax.set_xticks(range(len(betas)))
            ax.set_xticklabels([f"β={b:.2f}" for b in betas])
            ax.set_yticks(range(len(L_values)))
            ax.set_yticklabels([f"L={L}" for L in L_values])
            ax.set_xlabel("Temperature")
            ax.set_ylabel("Lattice size")
            ax.set_title(f"{model}: Same-L Test MSE")
            plt.colorbar(im, ax=ax)
            for i in range(len(L_values)):
                for j in range(len(betas)):
                    if not np.isnan(matrix[i, j]):
                        ax.text(j, i, f"{matrix[i,j]:.3f}",
                                ha="center", va="center", fontsize=9)
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"heatmap_{model}.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {FIG_DIR}/heatmap_{model}.png", flush=True)

    except Exception as e:
        print(f"\n  WARNING: Figure generation failed: {e}", flush=True)
        import traceback; traceback.print_exc()

    # ── RG Equivariance + Temperature Dependence ─────────────────────────────
    print("\n  === RG EQUIVARIANCE TEST ===", flush=True)
    eq_results = {}
    for seed in range(42, 42 + 5):
        torch.manual_seed(seed)
        np.random.seed(seed)
        ising = IsingModel(IsingConfig(L=16, beta=0.4407, h=0.0, J=1.0))
        ising.equilibriate(1000)
        rg = BlockSpinRG(block_size=2)
        fine, coarse1, coarse2 = [], [], []
        for _ in range(500 + 100):
            ising.metropolis_step(ising.state)
            s = ising.state.copy()
            fine.append(s)
            coarse1.append(rg.block_spin_transform(s))
            coarse2.append(rg.block_spin_transform(rg.block_spin_transform(s)))
        fine = np.array(fine[:500])
        coarse1 = np.array(coarse1[:500])
        coarse2 = np.array(coarse2[:500])

        fine_t = torch.from_numpy(fine.astype(np.float32))
        ds1 = SpinDataset(fine, coarse1)
        dl1 = DataLoader(ds1, batch_size=32, shuffle=True)

        # Train model on L→L/2
        model = FlatMLP(256, 64).to("cpu")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        for ep in range(200):
            model.train()
            for x, y in dl1:
                opt.zero_grad()
                loss = crit(model(x), y)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            # One-step: L→L/2
            mse1 = crit(model(fine_t), torch.from_numpy(coarse1.astype(np.float32))).item()
            # Two-step: L→L/2→L/4
            s1 = torch.tanh(model(fine_t)).numpy()
            s2 = np.array([rg.block_spin_transform(s1[i].reshape(8,8)).flatten()
                           for i in range(len(s1))])
            mse2 = float(crit(torch.from_numpy(s2.astype(np.float32)),
                               torch.from_numpy(coarse2.astype(np.float32))).item())
        eq_results[f"seed_{seed}"] = {"mse_L_to_L2": mse1, "mse_L_to_L4": mse2}
        print(f"    seed {seed}: MSE(L→L/2, flush=True)={mse1:.4f}, MSE(L→L/4)={mse2:.4f}, "
              f"Δ={abs(mse2-mse1):.4f}")

    with open(OUT_DIR / "rg_equivariance.json", "w") as f:
        json.dump(eq_results, f, indent=2)

    print("\n  === TEMPERATURE DEPENDENCE ===", flush=True)
    temp_dep = {}
    for beta in betas:
        print(f"\n  β={beta:.4f}:", flush=True)
        temp_dep[f"beta_{beta:.4f}"] = {}
        for model in ["MLP", "Linear"]:
            scores = []
            for seed in range(42, 42 + 5):
                _, mse = train_and_eval(
                    model, beta=beta,
                    n_train=500, n_test=300,
                    epochs=200, batch_size=32,
                    seed=seed, device="cpu",
                    L_data=16, L_target=16,
                    model_L_in=256, model_L_out=64,
                )
                scores.append(mse)
            temp_dep[f"beta_{beta:.4f}"][model] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores, ddof=1)),
            }
            print(f"    {model}: {np.mean(scores, flush=True):.4f} ± {np.std(scores,ddof=1):.4f}")

    with open(OUT_DIR / "temperature_dependence.json", "w") as f:
        json.dump(temp_dep, f, indent=2, default=str)

    summary = {
        "n_seeds": n_seeds, "n_runs": len(results),
        "wall_time_s": elapsed_total,
        "n_partA": total, "n_partB": total_b,
        "L_values": L_values, "betas": betas,
    }
    return df, stat_results, summary


if __name__ == "__main__":
    t0 = time.time()
    df, stats, summary = run_cross_scale_experiment(n_seeds=10)
    elapsed = time.time() - t0
    print(f"\n{'='*70}", flush=True)
    print(f"  COMPLETE in {elapsed:.0f}s ({elapsed/3600:.2f}h, flush=True)")
    print(f"{'='*70}", flush=True)
