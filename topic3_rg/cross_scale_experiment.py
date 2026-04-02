"""
Cross-Scale Transfer Experiment (R2) for Paper 3 RG × NN
=========================================================

本脚本现在与 `cross_scale_mlx.py` 对齐，采用同一套 patch-based transfer
protocol，避免再次生成与论文正文冲突的 zero-padding 旧结果。

设计:
  Part A — Same-L baselines (L ∈ {4,8,16}, β ∈ {0.30, βc, 0.60})
  Part B — Patch-based transfer proxy:
      训练:  L=8 原生输入 (64 dim) → 4×4 block-spin 输出 (16 dim)
      测试:  从 16×16 配置提取左上角 8×8 patch，再做同样的 64→16 评估
  Part C — RG Equivariance + Temperature Dependence (n=5 seeds)

输出:
  outputs/rg_bench/cross_scale/cross_scale_raw.csv
  outputs/rg_bench/cross_scale/cross_scale_statistics.json
  outputs/rg_bench/cross_scale/figures/*.png
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
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print(
        "ERROR: PyTorch is required for topic3_rg/cross_scale_experiment.py.\n"
        "This script now mirrors the MLX patch-based benchmark protocol, but it\n"
        "still requires a local torch install to run.\n"
        "If you want the benchmark used by the current paper revision, run\n"
        "topic3_rg/cross_scale_mlx.py in an MLX-capable session.",
        file=sys.stderr,
    )
    raise SystemExit(1)
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from topic3_rg.ising import IsingModel, IsingConfig, BlockSpinRG

OUT_DIR = Path(os.environ.get("RG_CROSS_SCALE_OUT_DIR", "outputs/rg_bench/cross_scale"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def configure_runtime_cache() -> None:
    root = Path(__file__).resolve().parent.parent
    mpl_dir = root / ".runtime-cache" / "matplotlib"
    xdg_dir = root / ".runtime-cache" / "xdg"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    xdg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_dir))


def print_protocol_banner(sampler: str = "wolff") -> None:
    print("\n" + "=" * 70, flush=True)
    print("  PAPER 3 MAIN BENCHMARK (PYTORCH MIRROR OF MLX PROTOCOL)", flush=True)
    print("  Protocol : patch-based transfer proxy", flush=True)
    print(f"  Sampler  : {sampler}", flush=True)
    print("  Part A   : same-L baselines on native lattices", flush=True)
    print("  Part B   : train on 8x8, test on 16x16 top-left 8x8 patch", flush=True)
    print(f"  Output   : {OUT_DIR}", flush=True)
    print("=" * 70, flush=True)


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


# ─── Training ────────────────────────────────────────────────────────────────

def train_and_eval(model_cls, beta, n_train, n_test, epochs, batch_size,
                   seed, device, L_data, L_target, model_L_in, model_L_out,
                   sampler: str = "wolff"):
    """
    训练并评估单个配置。
    当 `L_data=8, L_target=16, model_L_in=64` 时，使用 patch-based transfer:
    训练集来自原生 8×8 配置；测试集来自 16×16 配置的左上角 8×8 patch。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    rg = BlockSpinRG(block_size=2)

    # 训练数据始终在 L_data 原生尺度上生成
    ising = IsingModel(IsingConfig(L=L_data, beta=beta, h=0.0, J=1.0, sampler=sampler))
    ising.equilibriate(1000)
    fine_train, coarse_train = [], []

    for i in range(n_train + 200):
        ising.sampling_step(ising.state)
        fine = ising.state.copy()
        fine_train.append(fine.flatten())
        coarse = rg.block_spin_transform(fine.copy())
        coarse_train.append(coarse.flatten())
    fine_train = np.array(fine_train[:n_train])
    coarse_train = np.array(coarse_train[:n_train])

    # 测试数据在 L_target 上生成；patch-based transfer 时只抽取左上角 8×8 patch
    ising_test = IsingModel(IsingConfig(L=L_target, beta=beta, h=0.0, J=1.0, sampler=sampler))
    ising_test.equilibriate(1000)
    fine_test, coarse_test = [], []
    for _ in range(n_test + 100):
        ising_test.sampling_step(ising_test.state)
        fine = ising_test.state.copy()
        if model_L_in == 64 and L_target == 16:
            fine_in = fine[:8, :8]
            coarse = rg.block_spin_transform(fine_in.copy())
            fine_test.append(fine_in.flatten())
        else:
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
        # L = sqrt(model_L_in)
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
            pred = model(x_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()

    # Evaluate on L_target test data
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
                                epochs=200, batch_size=32, sampler: str = "wolff"):
    print("\n" + "="*70)
    print("  CROSS-SCALE TRANSFER EXPERIMENT (R2)")
    print(f"  {n_seeds} seeds, N_train={n_train}, epochs={epochs}")
    print(f"  sampler={sampler}")
    print("="*70)

    device = "cpu"
    betas = [0.30, 0.4407, 0.60]
    beta_labels = {0.30: "β=0.30 (disordered)", 0.4407: "β_c=0.4407 (critical)", 0.60: "β=0.60 (ordered)"}
    L_values = [4, 8, 16]

    seeds = list(range(42, 42 + n_seeds))
    results = []

    import csv

    # ── Checkpoint resume logic ──────────────────────────────────────────────
    ckpt_a = OUT_DIR / "cross_scale_partA.csv"
    ckpt_b = OUT_DIR / "cross_scale_partB.csv"

    # ── Part A: Same-L experiments ══════════════════════════════════════════
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

    total = len(part_a_configs)

    # Resume: skip Part A if checkpoint exists with all 360 rows
    if ckpt_a.exists():
        with open(ckpt_a) as f:
            reader = csv.DictReader(f)
            results = list(reader)
        for row in results:
            row.setdefault("sampler", "metropolis")
        sampler_matches = all(row.get("sampler") == sampler for row in results)
        if len(results) == total and sampler_matches:
            print(f"  [RESUME] Part A already complete ({len(results)} rows) — skipping")
        else:
            if not sampler_matches and results:
                print(f"  [RESUME] Part A sampler mismatch ({results[0].get('sampler')} != {sampler}) — recomputing from scratch")
            else:
                print(f"  [RESUME] Part A partial ({len(results)}/{total}) — recomputing from scratch")
            results = []
    else:
        results = []

    t0 = time.time()
    resumed_from = len(results)

    for i, cfg in enumerate(part_a_configs):
        if i < resumed_from:
            continue  # skip already done
        if (i+1) % 40 == 1:
            elapsed = time.time() - t0
            rate = (i+1 - resumed_from) / elapsed if elapsed > 0 else 0
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
            sampler=sampler,
        )
        results.append({**cfg, "train_mse": train_mse, "test_mse": test_mse,
                        "scale_distance": 0, "same_L": True, "part": "A", "sampler": sampler})

        # Checkpoint every 40 runs and at end of Part A
        if (i+1) % 40 == 0 or (i+1) == total:
            with open(ckpt_a, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"  Part A checkpoint ({len(results)}/{total} rows): {ckpt_a}", flush=True)

    part_a_results = results[:total]

    # ── Part B: Cross-scale experiment ════════════════════════════════════════
    print("\n\n[Part B] Patch-based transfer proxy: L=8 → L=16")
    print("  Train: native 8x8 input (64) -> 4x4 output (16)")
    print("  Test : top-left 8x8 patch extracted from each 16x16 config")
    print("  Models: MLP, Linear")

    part_b_configs = []
    for beta in betas:
        for model in ["MLP", "Linear"]:
            for seed in seeds:
                part_b_configs.append({
                    "L_data": 8, "L_target": 16,
                    "model_L_in": 64, "model_L_out": 16,
                    "beta": beta, "model": model, "seed": seed,
                })

    total_b = len(part_b_configs)

    # Resume: skip Part B if checkpoint exists with all 420 rows
    resumed_b_from = 0
    if ckpt_b.exists():
        with open(ckpt_b) as f:
            reader = csv.DictReader(f)
            part_b_results = list(reader)
        for row in part_b_results:
            row.setdefault("sampler", "metropolis")
        sampler_matches = all(row.get("sampler") == sampler for row in part_b_results)
        if len(part_b_results) == total + total_b and sampler_matches:
            results = part_b_results
            resumed_b_from = total_b
            print(f"  [RESUME] Part B already complete ({len(results)} total rows) — skipping to plots")
        elif len(part_b_results) > total and sampler_matches:
            resumed_b_from = len(part_b_results) - total
            results = part_a_results + part_b_results[total:]
            print(f"  [RESUME] Part B partial ({len(part_b_results)}/{total+total_b} total) — resuming from checkpoint")
        else:
            if not sampler_matches and part_b_results:
                print(f"  [RESUME] Part B sampler mismatch ({part_b_results[0].get('sampler')} != {sampler}); rebuilding Part B from fresh Part A results")
            else:
                print(f"  [RESUME] Part B checkpoint is incompatible ({len(part_b_results)}/{total+total_b}); rebuilding Part B from fresh Part A results")
            results = part_a_results.copy()
    else:
        results = part_a_results.copy()

    for i, cfg in enumerate(part_b_configs):
        if i < resumed_b_from:
            continue  # skip already done
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
            sampler=sampler,
        )
        results.append({**cfg, "train_mse": train_mse, "test_mse": test_mse,
                        "scale_distance": 8, "same_L": False, "part": "B", "sampler": sampler})

        # Checkpoint every 20 runs and at end of Part B
        if (i+1) % 20 == 0 or (i+1) == total_b:
            with open(ckpt_b, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"  Part B checkpoint ({len(results)} total rows): {ckpt_b}", flush=True)

    elapsed_total = time.time() - t0
    print(f"\n\n  Total: {len(results)} runs in {elapsed_total:.0f}s "
          f"({elapsed_total/len(results):.1f}s/run, "
          f"{elapsed_total/3600:.2f}h)")

    # ── Convert string values from CSV resume to proper types ──────────────────
    for r in results:
        # Boolean
        if r.get("same_L") in ("True", "False"):
            r["same_L"] = r["same_L"] == "True"
        # Numeric
        for _field in ["beta", "L_data", "L_target", "model_L_in", "model_L_out",
                       "train_mse", "test_mse", "scale_distance", "seed"]:
            if _field in r and isinstance(r[_field], str):
                try:
                    r[_field] = float(r[_field])
                except (ValueError, TypeError):
                    pass

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
                print(f"    MLP   : {np.mean(mlp_scores):.4f} ± {np.std(mlp_scores,ddof=1):.4f}")
                print(f"    Linear: {np.mean(lin_scores):.4f} ± {np.std(lin_scores,ddof=1):.4f}")
                print(f"    Welch t={t_stat:.3f}, p={t_pval:.4f}", flush=True)
                print(f"    Mann-Wh U={u_stat:.1f}, p={u_pval:.4f}", flush=True)
                print(f"    Permutation p={perm_p:.4f}, Cohen d={d:.3f}", flush=True)

    # Cross-scale comparison
    print("\n  Cross-scale (L=8→L=16) vs Same-L=16:")
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
                    "cross_mean": float(np.mean(cross_scores)),
                    "cross_std": float(np.std(cross_scores, ddof=1)),
                    "same_mean": float(np.mean(same_scores)),
                    "same_std": float(np.std(same_scores, ddof=1)),
                    "degradation": float(np.mean(cross_scores) / (np.mean(same_scores) + 1e-10)),
                }
                print(f"  {model} β={beta:.4f}: "
                      f"cross-scale={np.mean(cross_scores):.4f}±{np.std(cross_scores,ddof=1):.4f} "
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

        # Figure 2: Temperature dependence at L=16 for the two main baselines
        fig, ax = plt.subplots(figsize=(7, 5))
        beta_ticks = np.array(betas, dtype=float)
        for model, color in [("MLP", "steelblue"), ("Linear", "coral")]:
            means = []
            stds = []
            for beta in betas:
                grp = df_same[(df_same["L_data"] == 16) &
                              (df_same["beta"] == beta) &
                              (df_same["model"] == model)]["test_mse"]
                means.append(grp.mean() if len(grp) > 0 else np.nan)
                stds.append(grp.std(ddof=1) if len(grp) > 1 else 0.0)
            means = np.array(means, dtype=float)
            stds = np.array(stds, dtype=float)
            ax.plot(beta_ticks, means, marker="o", color=color, linewidth=2, label=model)
            ax.fill_between(beta_ticks, means - stds, means + stds, color=color, alpha=0.2)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel("Test MSE")
        ax.set_title("L=16 Temperature Dependence")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "temperature_dependence.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {FIG_DIR}/temperature_dependence.png", flush=True)

        # Figure 3: Cross-scale degradation
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
        plt.suptitle("Patch-Based Transfer Proxy: 8x8 Train vs 16x16-Patch Test (10 seeds)",
                     fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "cross_scale_degradation.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {FIG_DIR}/cross_scale_degradation.png", flush=True)

        # Figure 4: All same-L results heatmap
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

    # ── RG Equivariance + Temperature Dependence (skip on error) ──────────────
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
            print(f"    seed {seed}: MSE(L→L/2)={mse1:.4f}, MSE(L→L/4)={mse2:.4f}, "
                  f"Δ={abs(mse2-mse1):.4f}")

        with open(OUT_DIR / "rg_equivariance.json", "w") as f:
            json.dump(eq_results, f, indent=2)

    except Exception as e:
        print(f"  [RG equivariance skipped: {e}]", flush=True)

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
                print(f"    {model}: {np.mean(scores):.4f} ± {np.std(scores,ddof=1):.4f}")

        with open(OUT_DIR / "temperature_dependence.json", "w") as f:
            json.dump(temp_dep, f, indent=2, default=str)

    except Exception as e:
        print(f"  [Temperature dependence skipped: {e}]", flush=True)

    summary = {
        "n_seeds": n_seeds, "n_runs": len(results),
        "wall_time_s": elapsed_total,
        "n_partA": total, "n_partB": total_b,
        "L_values": L_values, "betas": betas,
        "sampler": sampler,
    }
    return df, stat_results, summary


if __name__ == "__main__":
    configure_runtime_cache()
    print_protocol_banner(sampler="wolff")
    t0 = time.time()
    df, stats, summary = run_cross_scale_experiment(n_seeds=10, sampler="wolff")
    elapsed = time.time() - t0
    print(f"\n{'='*70}", flush=True)
    print(f"  COMPLETE in {elapsed:.0f}s ({elapsed/3600:.2f}h)")
    print(f"{'='*70}", flush=True)
