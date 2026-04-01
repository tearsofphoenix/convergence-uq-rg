"""
Cross-Scale Transfer Experiment (R2) — MLX Implementation
======================================================
Pure MLX (Apple Silicon GPU).  Independent of PyTorch.

Definitive R2 experiment: can a model trained at L=8 generalize to L=16?

Design:
  Part A — Same-L baselines (L ∈ {4,8,16}, β ∈ {0.30, βc, 0.60}):
      MLP, Linear, CNN, RGMLP × 10 seeds → 3×3×4×10 = 360 runs

  Part B — Cross-scale transfer (core R2 test):
      Train at L=8 (input 64-dim, output 16-dim), test on L=16 top-left 8×8 patch
      (input 64-dim, output 16-dim).  Both MLP and Linear tested.

  Part C — RG Equivariance + Temperature Dependence (n=5 seeds)

Outputs:
  outputs/rg_bench/cross_scale/cross_scale_raw.csv
  outputs/rg_bench/cross_scale/cross_scale_statistics.json
  outputs/rg_bench/cross_scale/figures/*.png
"""
from __future__ import annotations

import sys, os, json, time
from pathlib import Path
from dataclasses import dataclass

import numpy as np

# ── MLX ────────────────────────────────────────────────────────────────────
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as opt
except ImportError:
    print("ERROR: MLX is required.  Install with: pip install mlx")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))
from topic3_rg.ising import IsingModel, IsingConfig, BlockSpinRG

OUT_DIR = Path("outputs/rg_bench/cross_scale")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

class FlatMLP(nn.Module):
    """Standard 3-hidden-layer MLP."""
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

    def __call__(self, x: mx.array) -> mx.array:
        return mx.tanh(self.net(x))


class LinearModel(nn.Module):
    """Single linear layer — minimal baseline."""
    def __init__(self, L_in: int = 256, L_out: int = 64):
        super().__init__()
        self.linear = nn.Linear(L_in, L_out)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.tanh(self.linear(x))


class CNNBlockSpin(nn.Module):
    """
    Block-spin CNN using reshape + Linear only (avoids Conv2d which triggers
    a tree_flatten bug in mx.value_and_grad on MLX 0.31.x).

    Forward pass:  [B, L, L] (channels-last)
        1. Reshape to [B, L//2, 2, L//2, 2] — group 2×2 blocks
        2. Transpose to [B, L//2, L//2, 2, 2]
        3. Reshape to [B, (L//2)^2, 4] — flatten spatial, 4 features per block
        4. Linear(4 → ch) per spatial position
        5. Linear(ch → 1) per spatial position
        6. Reshape to [B, (L//2)^2]
    """
    def __init__(self, L: int = 16, ch: int = 32):
        super().__init__()
        self.L = L
        self.ch = ch
        self.block_proj = nn.Linear(4, ch)  # 2×2 block → ch features
        self.h1 = nn.Linear(ch, ch)
        self.h2 = nn.Linear(ch, 1)

    def __call__(self, x: mx.array) -> mx.array:
        B = x.shape[0]
        L = self.L

        # Ensure [B, L, L]
        if x.ndim == 2:
            x = mx.reshape(x, [B, L, L])
        elif x.ndim == 3:
            x = mx.reshape(x, [B, L, L])

        # Pad if L is odd
        if L % 2 != 0:
            x = mx.pad(x, [[0, 0], [0, 1], [0, 1]])

        L2 = L // 2
        S2 = L2 * L2  # (L//2)^2 spatial positions

        # [B, L, L] → [B, L//2, 2, L//2, 2]
        x = mx.reshape(x, [B, L2, 2, L2, 2])
        # [B, L//2, 2, L//2, 2] → [B, L//2, L//2, 2, 2]
        x = mx.transpose(x, [0, 1, 3, 2, 4])
        # [B, L//2, L//2, 2, 2] → [B, S2, 4]
        x = mx.reshape(x, [B, S2, 4])

        # nn.Linear applies on last axis: [B, S2, 4] → [B, S2, ch]
        h = self.block_proj(x)
        h = nn.gelu(h)
        h = self.h1(h)
        h = nn.gelu(h)
        out = self.h2(h)        # [B, S2, 1]
        return mx.tanh(mx.reshape(out, [B, S2]))
class RGInformedMLP(nn.Module):
    """
    MLP whose first layer is pre-initialized to the correct block-sum.
    The encoder computes 2×2 block sums — the linear part of majority-vote.
    This is the 'knowledge-injected' baseline.
    """
    def __init__(self, L: int = 16, hidden: int = 256):
        super().__init__()
        new_L = L // 2
        # Build block-sum weight: each hidden unit averages one 2×2 block
        w = np.zeros((hidden, L * L), dtype=np.float32)
        for h in range(hidden):
            for bi in range(new_L):
                for bj in range(new_L):
                    for di in range(2):
                        for dj in range(2):
                            gi = (bi * 2 + di) % L
                            gj = (bj * 2 + dj) % L
                            w[h, gi * L + gj] = 0.25
        self.encoder = nn.Linear(L * L, hidden)
        self.encoder.weight = mx.array(w)
        self.encoder.bias = mx.zeros(hidden)
        self.rg_transform = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
        )
        self.decoder = nn.Linear(hidden // 2, new_L * new_L)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        h = self.rg_transform(self.encoder(x))
        return mx.tanh(self.decoder(h))


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class SpinDataset:
    """Lightweight NumPy dataset."""
    def __init__(self, fine: np.ndarray, coarse: np.ndarray):
        self.fine = fine.astype(np.float32)
        self.coarse = coarse.astype(np.float32)

    def __len__(self) -> int:
        return len(self.fine)

    def __getitem__(self, idx: int):
        return self.fine[idx], self.coarse[idx]


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(data: np.ndarray, stat=np.mean, n_boot: int = 10000, ci: float = 0.95):
    vals = np.array([
        stat(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    a = (1 - ci) / 2
    return float(np.percentile(vals, a * 100)), float(np.percentile(vals, (1 - a) * 100))


def permutation_test(g1: np.ndarray, g2: np.ndarray, n_perm: int = 10000):
    obs = float(np.mean(g1) - np.mean(g2))
    combined = np.concatenate([g1, g2])
    n1 = len(g1)
    cnt = 0
    for _ in range(n_perm):
        perm = np.random.permutation(combined)
        if float(np.mean(perm[:n1]) - np.mean(perm[n1:])) <= obs:
            cnt += 1
    return cnt / n_perm, obs


def cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    n1, n2 = len(g1), len(g2)
    s1, s2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled > 1e-10:
        return float((np.mean(g1) - np.mean(g2)) / pooled)
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN & EVAL
# ══════════════════════════════════════════════════════════════════════════════

def train_and_eval(
    model_cls: type,
    beta: float,
    n_train: int,
    n_test: int,
    epochs: int,
    batch_size: int,
    seed: int,
    L_data: int,
    L_target: int,
    model_L_in: int,
    model_L_out: int,
) -> tuple[float, float]:
    """
    Train and evaluate.  Returns (train_mse, test_mse).

    Cross-scale (L_data=8, L_target=16, model_L_in=64):
      - Train: generate L=8 configs (64-dim), coarse-grain at L=8 → 4×4=16-dim.
      - Test:  extract top-left 8×8 patch of each 16×16 test config (64-dim),
               coarse-grain that patch at L=8 → 4×4=16-dim.  Evaluate model on 64-dim input.
    """
    mx.random.seed(seed)
    np.random.seed(seed)
    rg = BlockSpinRG(block_size=2)

    # ── Training data ─────────────────────────────────────────────────────────
    ising = IsingModel(IsingConfig(L=L_data, beta=beta, h=0.0, J=1.0))
    ising.equilibriate(1000)
    fine_train, coarse_train = [], []

    for _ in range(n_train + 200):
        ising.metropolis_step(ising.state)
        fine = ising.state.copy()
        fine_train.append(fine.flatten())
        coarse = rg.block_spin_transform(fine)  # always at L_data
        coarse_train.append(coarse.flatten())

    fine_train = np.array(fine_train[:n_train])
    coarse_train = np.array(coarse_train[:n_train])

    # ── Test data ─────────────────────────────────────────────────────────────
    ising_t = IsingModel(IsingConfig(L=L_target, beta=beta, h=0.0, J=1.0))
    ising_t.equilibriate(1000)
    fine_test, coarse_test = [], []

    for _ in range(n_test + 100):
        ising_t.metropolis_step(ising_t.state)
        s = ising_t.state.copy()

        # Cross-scale Part B: extract top-left 8×8 patch of L_target=16 lattice
        if model_L_in == 64 and L_target == 16:
            s_in = s[:8, :8]                          # 8×8 = 64-dim input to model
            coarse = rg.block_spin_transform(s_in)      # 4×4 = 16-dim model output
            fine_test.append(s_in.flatten())            # 64-dim for cross-scale model
        else:
            coarse = rg.block_spin_transform(s)
            fine_test.append(s.flatten())              # normal: full lattice

        coarse_test.append(coarse.flatten())           # model output

    fine_test = np.array(fine_test[:n_test])
    coarse_test = np.array(coarse_test[:n_test])

    # ── Instantiate model ──────────────────────────────────────────────────────
    if model_cls == "MLP":
        model = FlatMLP(L_in=model_L_in, L_out=model_L_out)
    elif model_cls == "Linear":
        model = LinearModel(L_in=model_L_in, L_out=model_L_out)
    elif model_cls == "CNN":
        L_m = int(np.sqrt(model_L_in))
        model = CNNBlockSpin(L=L_m)
    elif model_cls == "RGMLP":
        L_m = int(np.sqrt(model_L_in))
        model = RGInformedMLP(L=L_m)
    else:
        raise ValueError(f"Unknown model: {model_cls}")

    # ── Prepare training batch data ────────────────────────────────────────────
    # Cross-scale: fine_train is already L_data=8 → 64-dim, matches model_L_in=64
    fine_train_np = fine_train

    optimizer = opt.Adam(learning_rate=1e-3)

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(epochs):
        perm = np.random.permutation(len(fine_train_np))
        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start:start + batch_size]
            x_b = mx.array(fine_train_np[batch_idx])
            y_b = mx.array(coarse_train[batch_idx])

            # MLX: mx.value_and_grad(fn)(model, x, y) returns (loss_value, grads_dict)
            def loss_fn(m, x, y):
                return mx.mean((m(x) - y) ** 2)

            _, grads = mx.value_and_grad(loss_fn)(model, x_b, y_b)
            # Clip global gradient norm (MLX 0.31.x: avoid NaN from exploding grads)
            def clip_grads(g, max_norm=1.0):
                def get_leaves(d):
                    for v in d.values():
                        if isinstance(v, dict):
                            yield from get_leaves(v)
                        elif isinstance(v, list):
                            for item in v:
                                if isinstance(item, dict):
                                    yield from get_leaves(item)
                                else:
                                    yield item
                        else:
                            yield v
                leaves = list(get_leaves(g))
                total_sq = sum(float((v * v).sum()) for v in leaves)
                g_norm = (total_sq + 1e-8) ** 0.5
                scale = min(1.0, max_norm / g_norm)
                if scale >= 1.0:
                    return g
                def scale_tree(d):
                    out = {}
                    for k, v in d.items():
                        if isinstance(v, dict):
                            out[k] = scale_tree(v)
                        elif isinstance(v, list):
                            out[k] = [item * scale if not isinstance(item, dict) else
                                      scale_tree(item) for item in v]
                        else:
                            out[k] = v * scale
                    return out
                return scale_tree(g)
            grads = clip_grads(grads)
            optimizer.update(model, grads)

    # Force evaluation
    mx.eval(list(model.trainable_parameters()))

    # ── Evaluate ─────────────────────────────────────────────────────────────
    train_x = mx.array(fine_train_np)
    train_y = mx.array(coarse_train)
    train_pred = model(train_x)
    mx.eval(train_pred)
    train_mse = float(mx.mean((train_pred - train_y) ** 2))

    # Test: use correct coarse labels (may be patch-based for cross-scale)
    test_x = mx.array(fine_test)
    test_y = mx.array(coarse_test)
    test_pred = model(test_x)
    mx.eval(test_pred)
    test_mse = float(mx.mean((test_pred - test_y) ** 2))

    return train_mse, test_mse


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(n_seeds: int = 10, n_train: int = 500, n_test: int = 300,
       epochs: int = 200, batch_size: int = 32):
    t0 = time.time()
    betas = [0.30, 0.4407, 0.60]
    beta_labels = {
        0.30: "beta=0.30 (disordered)",
        0.4407: "beta_c=0.4407 (critical)",
        0.60: "beta=0.60 (ordered)",
    }
    L_values = [4, 8, 16]
    seeds = list(range(42, 42 + n_seeds))
    results = []

    print("\n" + "=" * 70)
    print("  CROSS-SCALE TRANSFER EXPERIMENT (R2) — MLX")
    print(f"  {n_seeds} seeds, N_train={n_train}, epochs={epochs}")
    print(f"  Device: {mx.default_device()}")
    print("=" * 70)

    # ══ Part A: Same-L ════════════════════════════════════════════════════════
    print("\n[Part A] Same-L within-scale baselines")
    print("  Models: MLP, Linear, CNN, RGMLP  |  L in {4,8,16}, beta in {0.30,0.4407,0.60}")

    part_a = [
        {"L_data": L, "L_target": L,
         "model_L_in": L * L, "model_L_out": (L // 2) ** 2,
         "beta": beta, "model": model, "seed": seed}
        for L in L_values
        for beta in betas
        for model in ["MLP", "Linear", "CNN", "RGMLP"]
        for seed in seeds
    ]
    total_a = len(part_a)

    for i, cfg in enumerate(part_a):
        if (i + 1) % 40 == 1 or (i + 1) == total_a:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total_a - i - 1) / rate if rate > 0 else 0
            print(f"  A {i+1}/{total_a} ({100*(i+1)/total_a:.0f}%) "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

        tr_mse, te_mse = train_and_eval(
            model_cls=cfg["model"], beta=cfg["beta"],
            n_train=n_train, n_test=n_test, epochs=epochs, batch_size=batch_size,
            seed=cfg["seed"],
            L_data=cfg["L_data"], L_target=cfg["L_target"],
            model_L_in=cfg["model_L_in"], model_L_out=cfg["model_L_out"],
        )
        results.append({
            **cfg, "train_mse": tr_mse, "test_mse": te_mse,
            "scale_distance": 0, "same_L": True, "part": "A",
        })

    # Checkpoint A
    import csv as _csv
    with open(OUT_DIR / "cross_scale_partA.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"  Part A saved ({len(results)} rows)", flush=True)

    # ══ Part B: Cross-scale ══════════════════════════════════════════════════
    print("\n[Part B] Cross-scale: L=8 → L=16 transfer")
    print("  Model: L_in=64, L_out=16  (8×8 → 4×4 block)")
    print("  Test: evaluate on top-left 8×8 patch of 16×16 configs")

    part_b = [
        {"L_data": 8, "L_target": 16,
         "model_L_in": 64, "model_L_out": 16,
         "beta": beta, "model": model, "seed": seed}
        for beta in betas
        for model in ["MLP", "Linear"]
        for seed in seeds
    ]
    total_b = len(part_b)

    for i, cfg in enumerate(part_b):
        if (i + 1) % 20 == 1 or (i + 1) == total_b:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total_a + total_b - i - 1) / rate if rate > 0 else 0
            print(f"  B {i+1}/{total_b} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

        tr_mse, te_mse = train_and_eval(
            model_cls=cfg["model"], beta=cfg["beta"],
            n_train=n_train, n_test=n_test, epochs=epochs, batch_size=batch_size,
            seed=cfg["seed"],
            L_data=cfg["L_data"], L_target=cfg["L_target"],
            model_L_in=cfg["model_L_in"], model_L_out=cfg["model_L_out"],
        )
        results.append({
            **cfg, "train_mse": tr_mse, "test_mse": te_mse,
            "scale_distance": 8, "same_L": False, "part": "B",
        })

    # Checkpoint B
    with open(OUT_DIR / "cross_scale_partB.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"  Part B saved", flush=True)

    elapsed_total = time.time() - t0
    print(f"\n  {len(results)} runs in {elapsed_total:.0f}s "
          f"({elapsed_total/len(results):.1f}s/run, {elapsed_total/3600:.2f}h)")

    # ══ Raw CSV ══════════════════════════════════════════════════════════════════
    with open(OUT_DIR / "cross_scale_raw.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"  Raw CSV: {OUT_DIR / 'cross_scale_raw.csv'}", flush=True)

    # ══ Statistics ══════════════════════════════════════════════════════════════
    print("\n  === STATISTICS ===", flush=True)
    import pandas as pd
    from scipy import stats as scipy_stats

    df = pd.DataFrame(results)
    df_same = df[df["same_L"]].copy()
    df_cross = df[df["part"] == "B"].copy()
    stat = {}

    for L in L_values:
        for beta in betas:
            sub = df_same[(df_same["L_data"] == L) & (df_same["beta"] == beta)]
            for model in ["MLP", "Linear", "CNN", "RGMLP"]:
                grp = sub[sub["model"] == model]["test_mse"]
                if len(grp) > 0:
                    lo, hi = bootstrap_ci(grp.values)
                    key = f"L={L}_beta={beta:.4f}_{model}"
                    stat[key] = {
                        "L": int(L), "beta": float(beta), "model": model,
                        "mean": float(grp.mean()), "std": float(grp.std(ddof=1)),
                        "n": int(len(grp)), "ci_95": (float(lo), float(hi)),
                    }

            mlp_s = sub[sub["model"] == "MLP"]["test_mse"].values
            lin_s = sub[sub["model"] == "Linear"]["test_mse"].values
            if len(mlp_s) >= 2 and len(lin_s) >= 2:
                t_stat, t_p = scipy_stats.ttest_ind(mlp_s, lin_s, equal_var=False)
                u_stat, u_p = scipy_stats.mannwhitneyu(mlp_s, lin_s, alternative="two-sided")
                perm_p, obs_diff = permutation_test(mlp_s, lin_s)
                d = cohens_d(lin_s, mlp_s)
                key = f"L={L}_beta={beta:.4f}_MLPvsLinear"
                stat[key] = {
                    "MLP_mean": float(np.mean(mlp_s)), "MLP_std": float(np.std(mlp_s, ddof=1)),
                    "Linear_mean": float(np.mean(lin_s)), "Linear_std": float(np.std(lin_s, ddof=1)),
                    "Welch_t": float(t_stat), "Welch_p": float(t_p),
                    "MannWhitney_U": float(u_stat), "MannWhitney_p": float(u_p),
                    "Permutation_p": float(perm_p), "Cohens_d": float(d),
                    "obs_diff": float(obs_diff),
                }
                print(f"  L={L} beta={beta:.4f}: MLP={np.mean(mlp_s):.4f}±{np.std(mlp_s,ddof=1):.4f} "
                      f"Linear={np.mean(lin_s):.4f}±{np.std(lin_s,ddof=1):.4f} "
                      f"Welch p={float(t_p):.4f}", flush=True)

    # Cross-scale
    print("  Cross-scale (L=8→16) vs Same-L=16:", flush=True)
    for model in ["MLP", "Linear"]:
        for beta in betas:
            cs = df_cross[(df_cross["model"] == model) & (df_cross["beta"] == beta)]["test_mse"].values
            ss = df_same[(df_same["model"] == model) & (df_same["L_data"] == 16)
                       & (df_same["beta"] == beta)]["test_mse"].values
            if len(cs) >= 2 and len(ss) >= 2:
                key = f"CrossScale_{model}_beta={beta:.4f}"
                stat[key] = {
                    "cross_mean": float(np.mean(cs)), "cross_std": float(np.std(cs, ddof=1)),
                    "same_mean": float(np.mean(ss)), "same_std": float(np.std(ss, ddof=1)),
                    "degradation": float(np.mean(cs) / (np.mean(ss) + 1e-10)),
                }
                print(f"  {model} beta={beta:.4f}: cross={np.mean(cs):.4f}±{np.std(cs,ddof=1):.4f} "
                      f"vs same-L16={np.mean(ss):.4f}±{np.std(ss,ddof=1):.4f} "
                      f"ratio={np.mean(cs)/(np.mean(ss)+1e-10):.2f}x", flush=True)

    with open(OUT_DIR / "cross_scale_statistics.json", "w") as f:
        json.dump(stat, f, indent=2, default=str)
    print(f"  Statistics: {OUT_DIR / 'cross_scale_statistics.json'}", flush=True)

    # ══ Figures ══════════════════════════════════════════════════════════════════
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Figure 1: Same-L bars at L=16
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for col, beta in enumerate(betas):
            ax = axes[col]
            means, stds = [], []
            for model in ["MLP", "Linear", "CNN", "RGMLP"]:
                g = df_same[(df_same["L_data"]==16)&(df_same["beta"]==beta)
                           &(df_same["model"]==model)]["test_mse"]
                means.append(float(g.mean()) if len(g) else 0)
                stds.append(float(g.std(ddof=1)) if len(g) else 0)
            colors = ["steelblue", "coral", "forestgreen", "gold"]
            bars = ax.bar(["MLP","Linear","CNN","RGMLP"], means, yerr=stds, capsize=5,
                          color=colors, alpha=0.8)
            ax.set_ylabel("Test MSE"); ax.set_title(beta_labels[beta])
            ax.set_ylim(0, max(max(means)+max(stds)*2+0.1, 2.0))
            ax.grid(True, alpha=0.3, axis="y")
            for b, m, s in zip(bars, means, stds):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+s+0.02,
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
            s_means, s_stds, c_means, c_stds = [], [], [], []
            for model in ["MLP", "Linear"]:
                ss = df_same[(df_same["model"]==model)&(df_same["L_data"]==16)
                            &(df_same["beta"]==beta)]["test_mse"].values
                cs = df_cross[(df_cross["model"]==model)&(df_cross["beta"]==beta)]["test_mse"].values
                s_means.append(float(np.mean(ss)) if len(ss) else 0)
                s_stds.append(float(np.std(ss,ddof=1)) if len(ss) else 0)
                c_means.append(float(np.mean(cs)) if len(cs) else 0)
                c_stds.append(float(np.std(cs,ddof=1)) if len(cs) else 0)
            x = np.arange(2); w = 0.35
            ax.bar(x-w/2, s_means, w, yerr=s_stds, capsize=4,
                   label="Same-L=16", color="steelblue", alpha=0.8)
            ax.bar(x+w/2, c_means, w, yerr=c_stds, capsize=4,
                   label="Cross L=8→16", color="coral", alpha=0.8)
            ax.set_xticks(x); ax.set_xticklabels(["MLP","Linear"])
            ax.set_ylabel("Test MSE"); ax.set_title(beta_labels[beta])
            ax.legend(); ax.grid(True, alpha=0.3, axis="y")
        plt.suptitle("Cross-Scale Degradation: L=8→16 vs Same-L=16 (10 seeds)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "cross_scale_degradation.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {FIG_DIR}/cross_scale_degradation.png", flush=True)

        # Figure 3: Heatmaps
        for model in ["MLP", "Linear"]:
            fig, ax = plt.subplots(figsize=(6, 4))
            matrix = np.full((len(L_values), len(betas)), np.nan)
            for i, L in enumerate(L_values):
                for j, beta in enumerate(betas):
                    g = df_same[(df_same["L_data"]==L)&(df_same["beta"]==beta)
                               &(df_same["model"]==model)]["test_mse"]
                    if len(g): matrix[i, j] = float(g.mean())
            im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)
            ax.set_xticks(range(len(betas)))
            ax.set_xticklabels([f"β={b:.2f}" for b in betas])
            ax.set_yticks(range(len(L_values)))
            ax.set_yticklabels([f"L={L}" for L in L_values])
            ax.set_xlabel("Temperature"); ax.set_ylabel("Lattice size")
            ax.set_title(f"{model}: Same-L Test MSE")
            plt.colorbar(im, ax=ax)
            for i in range(len(L_values)):
                for j in range(len(betas)):
                    if not np.isnan(matrix[i, j]):
                        ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=9)
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"heatmap_{model}.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {FIG_DIR}/heatmap_{model}.png", flush=True)

    except Exception as e:
        print(f"\n  WARNING: figures failed: {e}", flush=True)
        import traceback; traceback.print_exc()

    # ══ Part C: RG Equivariance + Temperature Dependence ════════════════════════
    print("\n  === RG EQUIVARIANCE (Part C, n=5) ===", flush=True)
    eq = {}
    rg = BlockSpinRG(block_size=2)
    for seed in range(42, 42 + 5):
        mx.random.seed(seed); np.random.seed(seed)
        ising = IsingModel(IsingConfig(L=16, beta=0.4407, h=0.0, J=1.0))
        ising.equilibriate(1000)
        f1, c1, c2 = [], [], []
        for _ in range(500 + 100):
            ising.metropolis_step(ising.state)
            s = ising.state.copy()
            f1.append(s.flatten())                                      # 16×16 → 256
            c1.append(rg.block_spin_transform(s).flatten())             # 8×8  → 64
            c2.append(rg.block_spin_transform(rg.block_spin_transform(s)).flatten())  # 4×4 → 16
        f1 = np.array(f1[:500]); c1 = np.array(c1[:500]); c2 = np.array(c2[:500])

        model = FlatMLP(L_in=256, L_out=64)
        opt_ = opt.Adam(learning_rate=1e-3)
        perm = np.random.permutation(len(f1))
        for epoch in range(200):
            for st in range(0, len(perm), 32):
                bi = perm[st:st+32]
                x_b = mx.array(f1[bi]); y_b = mx.array(c1[bi])
                def loss_c(m, x, y):
                    return mx.mean((m(x) - y) ** 2)
                _, grads = mx.value_and_grad(loss_c)(model, x_b, y_b)
                # Clip grads
                def clip_grads(g, max_norm=1.0):
                    def get_leaves(d):
                        for v in d.values():
                            if isinstance(v, dict): yield from get_leaves(v)
                            elif isinstance(v, list):
                                for item in v:
                                    if isinstance(item, dict): yield from get_leaves(item)
                                    else: yield item
                            else: yield v
                    leaves = list(get_leaves(g))
                    total_sq = sum(float((v * v).sum()) for v in leaves)
                    g_norm = (total_sq + 1e-8) ** 0.5
                    scale = min(1.0, max_norm / g_norm)
                    if scale >= 1.0: return g
                    def scale_tree(d):
                        out = {}
                        for k, v in d.items():
                            if isinstance(v, dict): out[k] = scale_tree(v)
                            elif isinstance(v, list):
                                out[k] = [item * scale if not isinstance(item, dict)
                                           else scale_tree(item) for item in v]
                            else: out[k] = v * scale
                        return out
                    return scale_tree(g)
                grads = clip_grads(grads)
                opt_.update(model, grads)
        mx.eval(list(model.trainable_parameters()))

        f1_mx = mx.array(f1); c1_mx = mx.array(c1); c2_mx = mx.array(c2)
        mse1 = float(mx.mean((model(f1_mx) - c1_mx) ** 2))
        s1 = np.array(mx.tanh(model(f1_mx)))
        s2 = np.array([rg.block_spin_transform(s1[i].reshape(8,8)).flatten() for i in range(len(s1))])
        mse2 = float(np.mean((s2 - c2) ** 2))
        eq[f"seed_{seed}"] = {"mse_L_to_L2": mse1, "mse_L_to_L4": mse2}
        print(f"    seed {seed}: MSE(L→L/2)={mse1:.4f} MSE(L→L/4)={mse2:.4f} Δ={abs(mse2-mse1):.4f}")

    with open(OUT_DIR / "rg_equivariance.json", "w") as f:
        json.dump(eq, f, indent=2)

    print("\n  === TEMPERATURE DEPENDENCE (Part C, n=5) ===", flush=True)
    temp = {}
    for beta in betas:
        print(f"  beta={beta:.4f}:", flush=True)
        temp[f"beta_{beta:.4f}"] = {}
        for mname in ["MLP", "Linear"]:
            scores = []
            for seed in range(42, 42 + 5):
                _, mse = train_and_eval(
                    model_cls=mname, beta=beta,
                    n_train=500, n_test=300, epochs=200, batch_size=32,
                    seed=seed, L_data=16, L_target=16, model_L_in=256, model_L_out=64,
                )
                scores.append(mse)
            temp[f"beta_{beta:.4f}"][mname] = {
                "mean": float(np.mean(scores)), "std": float(np.std(scores, ddof=1)),
            }
            print(f"    {mname}: {np.mean(scores):.4f}±{np.std(scores,ddof=1):.4f}")

    with open(OUT_DIR / "temperature_dependence.json", "w") as f:
        json.dump(temp, f, indent=2, default=str)

    elapsed_total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  COMPLETE in {elapsed_total:.0f}s ({elapsed_total/3600:.2f}h)")
    print(f"{'='*70}")
    return df, stat, {"n_runs": len(results), "wall_time_s": elapsed_total,
                       "L_values": L_values, "betas": betas}


if __name__ == "__main__":
    run()
