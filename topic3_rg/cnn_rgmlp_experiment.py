"""
CNN vs RG-Informed-MLP vs Standard MLP Baseline Experiment (R2)
================================================================
Comparing: MLP, MLP-RGInit (FlatMLP with block-averaging init), CNNBlockSpin, RGInformedMLP
on Ising block-spin prediction at L ∈ {8, 16}, β ∈ {0.30, 0.4407, 0.60}

Design:
  - n_seeds=10 (seeds 42-51), N_train=500, N_test=300, epochs=200
  - At each (L,β): test MSE mean±std across seeds
  - Statistical tests: Welch t-test + Mann-Whitney U + permutation test (n_perm=5000)
    for every model pair at each (L,β)

Models:
  - FlatMLP(L_in=256, L_out=64, hidden=256): 3 hidden layers, GELU, tanh output
  - MLPWithRGInit: same architecture as FlatMLP but first-layer weights initialized
    to the block-averaging pattern
  - CNNBlockSpin(L=16): L-parametric CNN, output flattened to (L/2)^2
  - RGInformedMLP(L=16): MLP with block-average initialization in first layer

Note: For L=8, only MLP and MLP-RGInit can be trained (CNNBlockSpin and RGInformedMLP
require L=16 architecture). MLP models use L_in=64, L_out=16 at L=8.
For L=16, all 4 models are trained with L_in=256, L_out=64.

Outputs:
  outputs/rg_bench/cnn_rgmlp/cnn_rgmlp_raw.csv
  outputs/rg_bench/cnn_rgmlp/cnn_rgmlp_statistics.json
  outputs/rg_bench/cnn_rgmlp/cnn_rgmlp_summary.txt
"""
from __future__ import annotations
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

OUT_DIR = Path("/Users/isaacliu/workspace/convergence-uq-rg/outputs/rg_bench/cnn_rgmlp")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Models ──────────────────────────────────────────────────────────────────

class FlatMLP(nn.Module):
    """MLP with fixed input dim (L_in) and output dim (L_out)."""
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
        return torch.tanh(self.net(x))


def _make_block_averaging_weights(L_in: int, hidden: int) -> torch.Tensor:
    """Compute block-averaging initialization weights for first layer."""
    L = int(np.sqrt(L_in))
    new_L = L // 2
    w_init = torch.zeros(hidden, L_in)
    for h in range(hidden):
        for bi in range(new_L):
            for bj in range(new_L):
                k = (bi * new_L + bj) % hidden
                for di in range(2):
                    for dj in range(2):
                        gi = (bi * 2 + di) % L
                        gj = (bj * 2 + dj) % L
                        w_init[h, gi * L + gj] = 0.25
    return w_init * 4.0


class MLPWithRGInit(nn.Module):
    """
    FlatMLP architecture but with first-layer weights initialized to
    the block-averaging pattern (same as RGInformedMLP encoder).
    Architecture: Linear(L_in,hidden) -> GELU -> Linear(hidden,hidden) -> GELU ->
                   Linear(hidden,hidden//2) -> GELU -> Linear(hidden//2, L_out)
    """
    def __init__(self, L_in: int = 256, L_out: int = 64, hidden: int = 256):
        super().__init__()
        self.first_layer = nn.Linear(L_in, hidden)
        # Replace first-layer weights with block-averaging pattern
        with torch.no_grad():
            w_block = _make_block_averaging_weights(L_in, hidden)
            self.first_layer.weight.copy_(w_block)
            self.first_layer.bias.zero_()

        self.rest = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, L_out),
        )

    def forward(self, x):
        h = self.first_layer(x)
        return torch.tanh(self.rest(h))


class CNNBlockSpin(nn.Module):
    """CNN: L×L → L/2×L/2 block-spin. Works for any power-of-2 L ≥ 4."""
    def __init__(self, L: int = 16):
        super().__init__()
        self.L = L
        new_L = L // 2
        ch = max(8, L * 2)
        self.block_conv = nn.Conv2d(1, ch, kernel_size=2, stride=2)  # L→L/2
        self.net = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(ch, ch * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch * 2, 1, kernel_size=1),
        )

    def forward(self, x):
        if x.dim() == 2:
            L = int(np.sqrt(x.shape[1]))
            x = x.reshape(x.shape[0], L, L)
        h = x.unsqueeze(1)                          # [B,1,L,L]
        h = self.block_conv(h)                      # [B,ch,L/2,L/2]
        h = self.net(h)                             # [B,1,L/2,L/2]
        out = h.squeeze(1)                          # [B,L/2,L/2]
        return torch.tanh(out.reshape(x.shape[0], -1))  # [B, (L/2)^2]


class RGInformedMLP(nn.Module):
    """MLP with block-average initialization for L=16."""
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
        if x.dim() == 3:
            batch = x.shape[0]
            x = x.reshape(batch, -1)
        else:
            batch = x.shape[0]
        h = self.rg_transform(self.encoder(x))
        # Return flattened [B, (L/2)^2] to match target shape
        return torch.tanh(self.decoder(h).reshape(batch, -1))


# ─── Dataset ─────────────────────────────────────────────────────────────────

class SpinDataset(Dataset):
    def __init__(self, fine_configs, coarse_configs):
        self.fine = torch.from_numpy(fine_configs.astype(np.float32))
        self.coarse = torch.from_numpy(coarse_configs.astype(np.float32))
    def __len__(self):
        return len(self.fine)
    def __getitem__(self, idx):
        return self.fine[idx], self.coarse[idx]


# ─── Statistical tests ───────────────────────────────────────────────────────

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

def permutation_test(g1, g2, n_perm=5000):
    """Two-sided permutation test."""
    obs = np.mean(g1) - np.mean(g2)
    combined = np.concatenate([g1, g2])
    n1 = len(g1)
    cnt = 0
    for _ in range(n_perm):
        np.random.shuffle(combined)
        shuffled_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
        # Two-sided: count extreme in either direction
        if abs(shuffled_diff) >= abs(obs):
            cnt += 1
    return cnt / n_perm, float(obs)

def bootstrap_ci(data, stat=np.mean, n_boot=10000, ci=0.95):
    vals = [stat(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    a = (1 - ci) / 2
    return float(np.percentile(vals, a*100)), float(np.percentile(vals, (1-a)*100))


# ─── Training ─────────────────────────────────────────────────────────────────

def train_and_eval(model_cls, beta, n_train, n_test, epochs, batch_size,
                   seed, device, L):
    """
    Train model_cls at given beta and lattice size L.
    All data generated at scale L; model outputs (L/2)^2 block-spin prediction.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    rg = BlockSpinRG(block_size=2)

    # Generate training data at scale L
    ising = IsingModel(IsingConfig(L=L, beta=beta, h=0.0, J=1.0))
    ising.equilibriate(1000)
    fine_train, coarse_train = [], []
    for i in range(n_train + 200):
        ising.metropolis_step(ising.state)
        fine = ising.state.copy()
        coarse = rg.block_spin_transform(fine.copy())
        fine_train.append(fine.flatten())
        coarse_train.append(coarse.flatten())
    fine_train = np.array(fine_train[:n_train])
    coarse_train = np.array(coarse_train[:n_train])

    # Generate test data
    ising_test = IsingModel(IsingConfig(L=L, beta=beta, h=0.0, J=1.0))
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

    L_in = L * L
    L_out = (L // 2) * (L // 2)

    # Instantiate model
    if model_cls == "MLP":
        model = FlatMLP(L_in=L_in, L_out=L_out).to(device)
    elif model_cls == "MLPRGInit":
        model = MLPWithRGInit(L_in=L_in, L_out=L_out).to(device)
    elif model_cls == "CNN":
        model = CNNBlockSpin(L=L).to(device)
    elif model_cls == "RGMLP":
        model = RGInformedMLP(L=L).to(device)
    else:
        raise ValueError(f"Unknown model_cls: {model_cls}")

    train_ds = SpinDataset(fine_train, coarse_train)
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


# ─── Main experiment ───────────────────────────────────────────────────────────

def run_experiment(n_seeds=10, n_train=500, n_test=300, epochs=200, batch_size=32,
                   smoke_test=False):
    """
    Main experiment: CNN vs RG-MLP vs MLP baselines on Ising block-spin prediction.
    """
    print("\n" + "="*70)
    print("  CNN vs RGMLP vs MLP BASELINE EXPERIMENT")
    print(f"  n_seeds={n_seeds}, N_train={n_train}, N_test={n_test}, epochs={epochs}")
    print("="*70)

    device = "cpu"
    betas = [0.30, 0.4407, 0.60]
    beta_labels = {
        0.30: "β=0.30 (disordered)",
        0.4407: "β_c=0.4407 (critical)",
        0.60: "β=0.60 (ordered)"
    }
    L_values = [8, 16]

    # Models per L:
    #   L=8:  MLP, MLPRGInit
    #   L=16: MLP, MLPRGInit, CNN, RGMLP
    models_by_L = {
        8:  ["MLP", "MLPRGInit"],
        16: ["MLP", "MLPRGInit", "CNN", "RGMLP"],
    }

    if smoke_test:
        seeds = [42]
        print("\n  [SMOKE TEST MODE - 1 seed, all L/β/models]")
    else:
        seeds = list(range(42, 42 + n_seeds))

    results = []
    t0 = time.time()

    for L in L_values:
        for beta in betas:
            for model in models_by_L[L]:
                for seed in seeds:
                    print(f"\r  L={L}, β={beta:.4f}, {model}, seed={seed}...", 
                          end="", flush=True)
                    train_mse, test_mse = train_and_eval(
                        model_cls=model,
                        beta=beta,
                        n_train=n_train,
                        n_test=n_test,
                        epochs=epochs,
                        batch_size=batch_size,
                        seed=seed,
                        device=device,
                        L=L,
                    )
                    results.append({
                        "L": L,
                        "beta": beta,
                        "model": model,
                        "seed": seed,
                        "train_mse": train_mse,
                        "test_mse": test_mse,
                    })

    elapsed_total = time.time() - t0
    n_runs = len(results)
    print(f"\n\n  Total: {n_runs} runs in {elapsed_total:.0f}s "
          f"({elapsed_total/n_runs:.2f}s/run)")

    # ── Save raw results ───────────────────────────────────────────────────────
    import csv
    raw_path = OUT_DIR / "cnn_rgmlp_raw.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved: {raw_path}")

    # ── Statistical analysis ───────────────────────────────────────────────────
    print("\n  === STATISTICAL ANALYSIS ===")
    import pandas as pd
    df = pd.DataFrame(results)

    stat_results = {}
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("  CNN vs RGMLP vs MLP Baseline Experiment - Summary")
    summary_lines.append("=" * 70)
    summary_lines.append(f"  n_seeds={n_seeds if not smoke_test else 1}, "
                          f"N_train={n_train}, epochs={epochs}")
    summary_lines.append(f"  Total runs: {n_runs}")
    summary_lines.append(f"  Wall time: {elapsed_total:.0f}s")
    summary_lines.append("")

    for L in L_values:
        for beta in betas:
            sub = df[(df["L"] == L) & (df["beta"] == beta)]
            available_models = [m for m in models_by_L[L] if m in sub["model"].values]

            summary_lines.append(f"\n  L={L}, {beta_labels[beta]}")
            summary_lines.append("-" * 50)

            # Per-model summary stats
            model_stats = {}
            for model in available_models:
                grp = sub[sub["model"] == model]["test_mse"]
                if len(grp) > 0:
                    lo, hi = bootstrap_ci(grp.values)
                    model_stats[model] = {
                        "mean": float(grp.mean()),
                        "std": float(grp.std(ddof=1)),
                        "n": len(grp),
                        "ci_95": (float(lo), float(hi)),
                    }
                    summary_lines.append(
                        f"    {model:10s}: MSE={grp.mean():.4f} ± {grp.std(ddof=1):.4f} "
                        f"[95% CI: {lo:.4f}, {hi:.4f}]"
                    )

            # Pairwise statistical tests (all pairs)
            model_pairs = list(itertools.combinations(available_models, 2))
            for m1, m2 in model_pairs:
                g1 = sub[sub["model"] == m1]["test_mse"].values
                g2 = sub[sub["model"] == m2]["test_mse"].values
                if len(g1) < 2 or len(g2) < 2:
                    continue

                t_stat, t_pval = welch_ttest(g1, g2)
                u_stat, u_pval = mann_whitney_u(g1, g2)
                perm_p, obs_diff = permutation_test(g1, g2, n_perm=5000)
                d = cohens_d(g1, g2)

                key = f"L={L}_beta={beta:.4f}_{m1}_vs_{m2}"
                stat_results[key] = {
                    "L": L,
                    "beta": beta,
                    "model1": m1,
                    "model2": m2,
                    f"{m1}_mean": float(np.mean(g1)),
                    f"{m1}_std": float(np.std(g1, ddof=1)),
                    f"{m2}_mean": float(np.mean(g2)),
                    f"{m2}_std": float(np.std(g2, ddof=1)),
                    "Welch_t": t_stat,
                    "Welch_p": float(t_pval),
                    "MannWhitney_U": float(u_stat),
                    "MannWhitney_p": float(u_pval),
                    "Permutation_p": float(perm_p),
                    "Permutation_obs_diff": float(obs_diff),
                    "Cohens_d": d,
                }

                sig_marker = ""
                if t_pval < 0.05:
                    sig_marker += "*"
                if u_pval < 0.05:
                    sig_marker += "†"
                if perm_p < 0.05:
                    sig_marker += "‡"

                summary_lines.append(
                    f"    {m1} vs {m2}: Δ={obs_diff:+.4f}  "
                    f"Welch p={t_pval:.4f}{sig_marker}  "
                    f"MW p={u_pval:.4f}  Perm p={perm_p:.4f}  d={d:.3f}"
                )

    # Save statistics JSON
    stats_path = OUT_DIR / "cnn_rgmlp_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stat_results, f, indent=2, default=str)
    print(f"  Saved: {stats_path}")

    # Save human-readable summary
    summary_path = OUT_DIR / "cnn_rgmlp_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"  Saved: {summary_path}")

    # Also print summary
    for line in summary_lines:
        print(f"  {line}")

    print(f"\n{'='*70}")
    print(f"  COMPLETE in {elapsed_total:.0f}s ({elapsed_total/3600:.2f}h)")
    print(f"{'='*70}")

    return df, stat_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CNN vs RGMLP vs MLP baseline experiment")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run smoke test (1 seed, all models at L=8)")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of seeds (default: 10)")
    parser.add_argument("--train", type=int, default=500,
                        help="N_train (default: 500)")
    parser.add_argument("--test", type=int, default=300,
                        help="N_test (default: 300)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Epochs (default: 200)")
    args = parser.parse_args()

    t0 = time.time()
    df, stats = run_experiment(
        n_seeds=args.seeds,
        n_train=args.train,
        n_test=args.test,
        epochs=args.epochs,
        smoke_test=args.smoke_test,
    )
    elapsed = time.time() - t0
    print(f"\nTotal script time: {elapsed:.0f}s")
