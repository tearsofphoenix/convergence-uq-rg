"""
Topic 3 RG Benchmark: MLP vs Linear Comparison
================================================
Trains FlatMLP, LinearModel, CNNBlockSpin, RGInformedMLP at each (L, beta).
Computes per-model statistics and MLP vs Linear statistical comparisons.

Outputs:
  outputs/rg_bench/rg_statistics.json  — all statistical results
  outputs/rg_bench/figures/           — comparison plots
"""
from __future__ import annotations
import sys, os, json, time, itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from topic3_rg.ising import IsingModel, IsingConfig, BlockSpinRG

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import stats

OUT_DIR = Path("/Users/isaacliu/workspace/convergence-uq-rg/outputs/rg_bench")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─── Models (flattened output) ────────────────────────────────────────────────

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
    """CNN: L×L → L/2×L/2 block-spin. Works for any power-of-2 L ≥ 4.
    Output is flattened [B, (L/2)^2].
    """
    def __init__(self, L: int = 16):
        super().__init__()
        self.L = L
        new_L = L // 2
        ch = max(8, L * 2)
        self.block_conv = nn.Conv2d(1, ch, kernel_size=2, stride=2)
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
        h = x.unsqueeze(1)
        h = self.block_conv(h)
        h = self.net(h)
        out = h.squeeze(1)
        # Flatten to [B, (L/2)^2]
        return torch.tanh(out.reshape(x.shape[0], -1))


class RGInformedMLP(nn.Module):
    """MLP with block-average initialization. Output flattened [B, (L/2)^2].
    FIXED: output is now flattened instead of 2D.
    """
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
        # Flattened output [B, (L/2)^2] instead of 2D
        return torch.tanh(self.decoder(h).reshape(batch, -1))


# ─── Dataset ──────────────────────────────────────────────────────────────────

class SpinDataset(Dataset):
    def __init__(self, fine_configs, coarse_configs):
        self.fine = torch.from_numpy(fine_configs.astype(np.float32))
        self.coarse = torch.from_numpy(coarse_configs.astype(np.float32))
    def __len__(self):
        return len(self.fine)
    def __getitem__(self, idx):
        return self.fine[idx], self.coarse[idx]


# ─── Training ─────────────────────────────────────────────────────────────────

def train_and_eval(model_cls, beta, n_train, n_test, epochs, batch_size,
                   seed, device, L_data, L_target, model_L_in, model_L_out):
    """Train model_cls at given beta. Returns (train_mse, test_mse)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    rg = BlockSpinRG(block_size=2)

    # Generate training data
    ising = IsingModel(IsingConfig(L=L_data, beta=beta, h=0.0, J=1.0))
    ising.equilibriate(1000)
    fine_train, coarse_train = [], []

    for i in range(n_train + 200):
        ising.metropolis_step(ising.state)
        fine = ising.state.copy()
        fine_train.append(fine.flatten())
        coarse = rg.block_spin_transform(fine.copy())
        coarse_train.append(coarse.flatten())
    fine_train = np.array(fine_train[:n_train])
    coarse_train = np.array(coarse_train[:n_train])

    # Generate test data at L_target scale
    ising_test = IsingModel(IsingConfig(L=L_target, beta=beta, h=0.0, J=1.0))
    ising_test.equilibriate(1000)
    fine_test, coarse_test = [], []
    for _ in range(n_test + 100):
        ising_test.metropolis_step(ising_test.state)
        fine = ising_test.state.copy()
        coarse = rg.block_spin_transform(fine.copy())
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
        L_model = int(np.sqrt(model_L_in))
        model = CNNBlockSpin(L=L_model)
    elif model_cls == "RGMLP":
        L_model = int(np.sqrt(model_L_in))
        model = RGInformedMLP(L=L_model)
    else:
        raise ValueError(model_cls)
    model = model.to(device)

    train_ds = SpinDataset(fine_train, coarse_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(x_b)  # pred is [B, model_L_out] flattened
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        fine_t = torch.from_numpy(fine_test.astype(np.float32)).to(device)
        coarse_t = torch.from_numpy(coarse_test.astype(np.float32)).to(device)
        pred_test = model(fine_t)
        test_mse = criterion(pred_test, coarse_t).item()

        fine_t_train = torch.from_numpy(fine_train.astype(np.float32)).to(device)
        coarse_t_train = torch.from_numpy(coarse_train.astype(np.float32)).to(device)
        pred_train = model(fine_t_train)
        train_mse = criterion(pred_train, coarse_t_train).item()

    return float(train_mse), float(test_mse)


# ─── Statistical tests ────────────────────────────────────────────────────────

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
    """Two-sided permutation test. Returns (p_value, observed_diff)."""
    obs = np.mean(g1) - np.mean(g2)
    combined = np.concatenate([g1, g2])
    n1 = len(g1)
    cnt = 0
    for _ in range(n_perm):
        np.random.shuffle(combined)
        perm_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
        # Two-sided: count extreme in either direction
        if abs(perm_diff) >= abs(obs):
            cnt += 1
    return float(cnt / n_perm), float(obs)


def bootstrap_ci(data, stat=np.mean, n_boot=10000, ci=0.95):
    """Percentile bootstrap CI for given statistic."""
    vals = [stat(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    a = (1 - ci) / 2
    return float(np.percentile(vals, a * 100)), float(np.percentile(vals, (1 - a) * 100))


def compute_per_model_stats(scores):
    """Compute mean, std, median, bootstrap CI, min, max for a list of scores."""
    scores = np.array(scores)
    lo, hi = bootstrap_ci(scores)
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)),
        "median": float(np.median(scores)),
        "ci_95": (lo, hi),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "n": len(scores),
    }


# ─── Main experiment ──────────────────────────────────────────────────────────

def run_rg_benchmark(n_seeds=10, n_train=500, n_test=300,
                     epochs=200, batch_size=32):
    print("\n" + "=" * 70)
    print("  RG BENCHMARK: MLP vs Linear Comparison")
    print(f"  {n_seeds} seeds, N_train={n_train}, epochs={epochs}")
    print("=" * 70)

    device = "cpu"
    betas = [0.30, 0.4407, 0.60]
    beta_labels = {0.30: "disordered", 0.4407: "critical", 0.60: "ordered"}
    L_values = [4, 8, 16]
    model_types = ["MLP", "Linear", "CNN", "RGMLP"]

    seeds = list(range(42, 42 + n_seeds))
    results = []

    # ── Run all (L, beta, model, seed) combinations ────────────────────────────
    configs = []
    for L in L_values:
        for beta in betas:
            for model in model_types:
                for seed in seeds:
                    configs.append({
                        "L": L, "beta": beta, "model": model, "seed": seed,
                        "L_data": L, "L_target": L,
                        "model_L_in": L * L, "model_L_out": (L // 2) * (L // 2),
                    })

    t0 = time.time()
    total = len(configs)
    print(f"\n  Total configurations: {total}")

    for i, cfg in enumerate(configs):
        if (i + 1) % 40 == 1 or i == total - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  Progress: {i+1}/{total} ({100*(i+1)/total:.1f}%) "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

        train_mse, test_mse = train_and_eval(
            model_cls=cfg["model"], beta=cfg["beta"],
            n_train=n_train, n_test=n_test,
            epochs=epochs, batch_size=batch_size,
            seed=cfg["seed"], device=device,
            L_data=cfg["L_data"], L_target=cfg["L_target"],
            model_L_in=cfg["model_L_in"], model_L_out=cfg["model_L_out"],
        )
        results.append({
            "L": cfg["L"], "beta": cfg["beta"], "model": cfg["model"],
            "seed": cfg["seed"], "train_mse": train_mse, "test_mse": test_mse,
        })

    elapsed_total = time.time() - t0
    print(f"\n  Total: {len(results)} runs in {elapsed_total:.0f}s "
          f"({elapsed_total/len(results):.2f}s/run)")

    # ── Save raw results ─────────────────────────────────────────────────────
    import csv
    df_path = OUT_DIR / "rg_raw.csv"
    with open(df_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved: {df_path}")

    # ── Statistical analysis ─────────────────────────────────────────────────
    print("\n  === STATISTICAL ANALYSIS ===")
    stat_results = {}

    import pandas as pd
    df = pd.DataFrame(results)

    # Per-model stats: for each (L, beta, model) compute descriptive stats
    for L in L_values:
        for beta in betas:
            for model in model_types:
                grp = df[(df["L"] == L) & (df["beta"] == beta) &
                         (df["model"] == model)]["test_mse"]
                if len(grp) >= 2:
                    key = f"L={L}_beta={beta:.4f}_{model}"
                    stat_results[key] = {
                        "L": L, "beta": beta, "model": model,
                        **compute_per_model_stats(grp.values),
                    }

    # MLP vs Linear comparison at each (L, beta)
    print("\n  MLP vs Linear comparisons:")
    for L in L_values:
        for beta in betas:
            mlp_scores = df[(df["L"] == L) & (df["beta"] == beta) &
                            (df["model"] == "MLP")]["test_mse"].values
            lin_scores = df[(df["L"] == L) & (df["beta"] == beta) &
                            (df["model"] == "Linear")]["test_mse"].values

            if len(mlp_scores) >= 2 and len(lin_scores) >= 2:
                t_stat, t_pval = welch_ttest(mlp_scores, lin_scores)
                u_stat, u_pval = mann_whitney_u(mlp_scores, lin_scores)
                perm_p, obs_diff = permutation_test(mlp_scores, lin_scores, n_perm=10000)
                d = cohens_d(mlp_scores, lin_scores)

                key = f"L={L}_beta={beta:.4f}_MLPvsLinear"
                stat_results[key] = {
                    "L": L, "beta": beta,
                    "MLP_mean": float(np.mean(mlp_scores)),
                    "MLP_std": float(np.std(mlp_scores, ddof=1)),
                    "MLP_median": float(np.median(mlp_scores)),
                    "MLP_ci_95": bootstrap_ci(mlp_scores),
                    "Linear_mean": float(np.mean(lin_scores)),
                    "Linear_std": float(np.std(lin_scores, ddof=1)),
                    "Linear_median": float(np.median(lin_scores)),
                    "Linear_ci_95": bootstrap_ci(lin_scores),
                    "Welch_t": t_stat, "Welch_p": t_pval,
                    "MannWhitney_U": u_stat, "MannWhitney_p": u_pval,
                    "Permutation_p": perm_p, "Permutation_n_perm": 10000,
                    "Cohens_d": d,
                    "obs_diff_MLP_minus_Linear": obs_diff,
                    "n_MLP": len(mlp_scores), "n_Linear": len(lin_scores),
                }

                print(f"\n  L={L}, β={beta:.4f} ({beta_labels[beta]}):")
                print(f"    MLP   : {np.mean(mlp_scores):.4f} ± {np.std(mlp_scores,ddof=1):.4f} "
                      f"[median={np.median(mlp_scores):.4f}]")
                print(f"    Linear: {np.mean(lin_scores):.4f} ± {np.std(lin_scores,ddof=1):.4f} "
                      f"[median={np.median(lin_scores):.4f}]")
                print(f"    Welch t={t_stat:.3f}, p={t_pval:.4f}")
                print(f"    Mann-Wh U={u_stat:.1f}, p={u_pval:.4f}")
                print(f"    Permutation p={perm_p:.4f} (n_perm=10000), Cohen d={d:.3f}")

    # ── Save statistics JSON ─────────────────────────────────────────────────
    stats_path = OUT_DIR / "rg_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stat_results, f, indent=2, default=str)
    print(f"\n  Saved: {stats_path}")

    # ── Figures ──────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Figure 1: Bar chart comparing all models at each beta for L=16
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for col, beta in enumerate(betas):
            ax = axes[col]
            means, stds, labels = [], [], []
            for model in model_types:
                grp = df[(df["L"] == 16) & (df["beta"] == beta) &
                         (df["model"] == model)]["test_mse"]
                if len(grp) > 0:
                    means.append(grp.mean())
                    stds.append(grp.std(ddof=1))
                    labels.append(model)
            colors = ["steelblue", "coral", "forestgreen", "gold"]
            bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors[:len(means)], alpha=0.8)
            ax.set_ylabel("Test MSE")
            ax.set_title(f"β={beta:.4f} ({beta_labels[beta]})")
            ax.grid(True, alpha=0.3, axis="y")
            for bar, m, s in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.02,
                        f"{m:.3f}", ha="center", va="bottom", fontsize=9)
        plt.suptitle("L=16: Model Comparison by Temperature (10 seeds)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "model_comparison_L16.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {FIG_DIR}/model_comparison_L16.png")

        # Figure 2: MLP vs Linear comparison heatmap
        for model_pair in ["MLP", "Linear"]:
            fig, ax = plt.subplots(figsize=(8, 6))
            matrix = np.full((len(L_values), len(betas)), np.nan)
            for i, L in enumerate(L_values):
                for j, beta in enumerate(betas):
                    grp = df[(df["L"] == L) & (df["beta"] == beta) &
                             (df["model"] == model_pair)]["test_mse"]
                    if len(grp) > 0:
                        matrix[i, j] = grp.mean()
            im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
            ax.set_xticks(range(len(betas)))
            ax.set_xticklabels([f"β={b:.2f}" for b in betas])
            ax.set_yticks(range(len(L_values)))
            ax.set_yticklabels([f"L={L}" for L in L_values])
            ax.set_xlabel("Temperature")
            ax.set_ylabel("Lattice size")
            ax.set_title(f"{model_pair}: Test MSE Heatmap")
            plt.colorbar(im, ax=ax)
            for i in range(len(L_values)):
                for j in range(len(betas)):
                    if not np.isnan(matrix[i, j]):
                        ax.text(j, i, f"{matrix[i,j]:.3f}",
                                ha="center", va="center", fontsize=10)
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"heatmap_{model_pair}.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {FIG_DIR}/heatmap_{model_pair}.png")

        # Figure 3: MLP vs Linear effect size (Cohen's d) across conditions
        fig, ax = plt.subplots(figsize=(8, 6))
        d_matrix = np.full((len(L_values), len(betas)), np.nan)
        for i, L in enumerate(L_values):
            for j, beta in enumerate(betas):
                key = f"L={L}_beta={beta:.4f}_MLPvsLinear"
                if key in stat_results:
                    d_matrix[i, j] = stat_results[key]["Cohens_d"]
        im = ax.imshow(d_matrix, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
        ax.set_xticks(range(len(betas)))
        ax.set_xticklabels([f"β={b:.2f}" for b in betas])
        ax.set_yticks(range(len(L_values)))
        ax.set_yticklabels([f"L={L}" for L in L_values])
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Lattice size")
        ax.set_title("Cohen's d: MLP vs Linear (positive=MLP better)")
        plt.colorbar(im, ax=ax)
        for i in range(len(L_values)):
            for j in range(len(betas)):
                if not np.isnan(d_matrix[i, j]):
                    ax.text(j, i, f"{d_matrix[i,j]:.2f}",
                            ha="center", va="center", fontsize=10)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "cohens_d_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {FIG_DIR}/cohens_d_heatmap.png")

    except Exception as e:
        print(f"\n  WARNING: Figure generation failed: {e}")
        import traceback; traceback.print_exc()

    summary = {
        "n_seeds": n_seeds, "n_runs": len(results),
        "wall_time_s": elapsed_total,
        "L_values": L_values, "betas": betas, "models": model_types,
    }
    return df, stat_results, summary


if __name__ == "__main__":
    t0 = time.time()
    df, stats, summary = run_rg_benchmark(n_seeds=10)
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  COMPLETE in {elapsed:.0f}s ({elapsed/3600:.2f}h)")
    print(f"  Statistics saved to: {OUT_DIR / 'rg_statistics.json'}")
    print(f"{'='*70}")
