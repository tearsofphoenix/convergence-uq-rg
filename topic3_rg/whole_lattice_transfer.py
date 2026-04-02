"""
Whole-lattice and multi-scale transfer experiments for Paper 3.

核心思想:
  - 在 source lattice 上训练局部 block-spin 预测器
  - 对 target lattice 进行非重叠 patch 切分
  - 将训练好的局部模型平铺到整张 target lattice 上
  - 组装出完整 coarse lattice，从而做真正的 whole-lattice transfer

输出:
  outputs/rg_bench/whole_lattice_transfer/raw.csv
  outputs/rg_bench/whole_lattice_transfer/summary.json
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from topic3_rg.ising import BlockSpinRG, IsingConfig, IsingModel
from topic3_rg.cross_scale_experiment import FlatMLP, LinearModel


OUT_DIR = Path(os.environ.get("RG_WHOLE_LATTICE_OUT_DIR", "outputs/rg_bench/whole_lattice_transfer"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"
RG = BlockSpinRG(block_size=2)


def generate_dataset(L: int, beta: float, n_samples: int, seed: int, eq_steps: int = 1000,
                     sampler: str = "wolff") -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    ising = IsingModel(IsingConfig(L=L, beta=beta, h=0.0, J=1.0, sampler=sampler))
    ising.equilibriate(eq_steps)

    fine, coarse = [], []
    for _ in range(n_samples + 200):
        ising.sampling_step(ising.state)
        s = ising.state.copy()
        fine.append(s.astype(np.float32).reshape(-1))
        coarse.append(RG.block_spin_transform(s).astype(np.float32).reshape(-1))
    return np.array(fine[:n_samples]), np.array(coarse[:n_samples])


def build_model(model_name: str, L_in: int, L_out: int) -> nn.Module:
    if model_name == "MLP":
        return FlatMLP(L_in=L_in, L_out=L_out)
    if model_name == "Linear":
        return LinearModel(L_in=L_in, L_out=L_out)
    raise ValueError(model_name)


def train_source_model(model_name: str, source_L: int, beta: float, seed: int,
                       n_train: int, epochs: int, batch_size: int,
                       sampler: str = "wolff") -> tuple[nn.Module, float]:
    fine_train, coarse_train = generate_dataset(source_L, beta, n_train, seed, sampler=sampler)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(fine_train), torch.from_numpy(coarse_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    model = build_model(model_name, source_L * source_L, (source_L // 2) ** 2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x_b), y_b)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        train_pred = model(torch.from_numpy(fine_train).to(DEVICE))
        train_mse = crit(train_pred, torch.from_numpy(coarse_train).to(DEVICE)).item()
    return model, float(train_mse)


def apply_model_whole_lattice(model: nn.Module, source_L: int, fine_batch: np.ndarray, target_L: int) -> np.ndarray:
    batch = fine_batch.shape[0]
    if source_L == target_L:
        with torch.no_grad():
            pred = model(torch.from_numpy(fine_batch.reshape(batch, -1).astype(np.float32)).to(DEVICE))
        return pred.detach().cpu().numpy()

    patches_per_dim = target_L // source_L
    coarse_patch_L = source_L // 2
    target_coarse_L = target_L // 2
    pred_full = np.zeros((batch, target_coarse_L, target_coarse_L), dtype=np.float32)

    with torch.no_grad():
        for pi in range(patches_per_dim):
            for pj in range(patches_per_dim):
                patch = fine_batch[
                    :,
                    pi * source_L:(pi + 1) * source_L,
                    pj * source_L:(pj + 1) * source_L,
                ].reshape(batch, -1).astype(np.float32)
                pred_patch = model(torch.from_numpy(patch).to(DEVICE)).detach().cpu().numpy()
                pred_patch = pred_patch.reshape(batch, coarse_patch_L, coarse_patch_L)
                pred_full[
                    :,
                    pi * coarse_patch_L:(pi + 1) * coarse_patch_L,
                    pj * coarse_patch_L:(pj + 1) * coarse_patch_L,
                ] = pred_patch

    return pred_full.reshape(batch, -1)


def evaluate_on_target(model: nn.Module, source_L: int, target_L: int, beta: float,
                       seed: int, n_test: int, sampler: str = "wolff") -> float:
    np.random.seed(seed + 10000)
    torch.manual_seed(seed + 10000)
    ising = IsingModel(IsingConfig(L=target_L, beta=beta, h=0.0, J=1.0, sampler=sampler))
    ising.equilibriate(1000)

    fine, coarse = [], []
    for _ in range(n_test + 100):
        ising.sampling_step(ising.state)
        s = ising.state.copy()
        fine.append(s.astype(np.float32))
        coarse.append(RG.block_spin_transform(s).astype(np.float32).reshape(-1))
    fine = np.array(fine[:n_test], dtype=np.float32)
    coarse = np.array(coarse[:n_test], dtype=np.float32)

    pred = apply_model_whole_lattice(model, source_L, fine, target_L)
    mse = float(np.mean((pred - coarse) ** 2))
    return mse


def summarize(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    combos = {}
    for row in rows:
        key = (row["source_L"], row["target_L"], row["beta"], row["model"])
        combos.setdefault(key, []).append(row["test_mse"])

    same_l = {}
    for (source_L, target_L, beta, model), values in combos.items():
        if source_L == target_L:
            same_l[(target_L, beta, model)] = float(np.mean(values))

    for key, values in combos.items():
        source_L, target_L, beta, model = key
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        base = same_l.get((target_L, beta, model), np.nan)
        summary[f"L{source_L}_to_L{target_L}_{model}_beta_{beta:.4f}"] = {
            "source_L": source_L,
            "target_L": target_L,
            "beta": beta,
            "model": model,
            "sampler": rows[0].get("sampler", "unknown"),
            "mean": mean,
            "std": std,
            "n": len(values),
            "degradation_vs_same_target": float(mean / base) if np.isfinite(base) and base > 0 else None,
        }
    return summary


def run_experiment(n_seeds: int = 10, n_train: int = 500, n_test: int = 300,
                   epochs: int = 200, batch_size: int = 32,
                   sampler: str = "wolff") -> tuple[list[dict], dict]:
    print("\n" + "=" * 70, flush=True)
    print("  WHOLE-LATTICE / MULTI-SCALE TRANSFER EXPERIMENT", flush=True)
    print("  Protocol : tile source-scale predictor over the full target lattice", flush=True)
    print(f"  Sampler  : {sampler}", flush=True)
    print(f"  Output   : {OUT_DIR}", flush=True)
    print("=" * 70, flush=True)

    betas = [0.30, 0.4407, 0.60]
    source_scales = [4, 8, 16]
    target_map = {4: [4, 8, 16], 8: [8, 16], 16: [16]}
    models = ["MLP", "Linear"]
    seeds = list(range(42, 42 + n_seeds))

    rows: list[dict] = []
    total_trainings = len(source_scales) * len(betas) * len(models) * len(seeds)
    train_idx = 0
    t0 = time.time()

    for source_L in source_scales:
        for beta in betas:
            for model_name in models:
                for seed in seeds:
                    train_idx += 1
                    if train_idx == 1 or train_idx % 10 == 0:
                        elapsed = time.time() - t0
                        rate = train_idx / elapsed if elapsed > 0 else 0.0
                        eta = (total_trainings - train_idx) / rate if rate > 0 else 0.0
                        print(
                            f"  train {train_idx}/{total_trainings}: "
                            f"{model_name} source_L={source_L} beta={beta:.4f} "
                            f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                            flush=True,
                        )

                    model, train_mse = train_source_model(
                        model_name, source_L, beta, seed, n_train, epochs, batch_size, sampler=sampler
                    )

                    for target_L in target_map[source_L]:
                        test_mse = evaluate_on_target(model, source_L, target_L, beta, seed, n_test, sampler=sampler)
                        rows.append({
                            "source_L": source_L,
                            "target_L": target_L,
                            "beta": beta,
                            "model": model_name,
                            "sampler": sampler,
                            "seed": seed,
                            "train_mse": train_mse,
                            "test_mse": test_mse,
                            "whole_lattice": True,
                        })

    summary = summarize(rows)

    raw_path = OUT_DIR / "raw.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  saved: {raw_path}", flush=True)
    print(f"  saved: {summary_path}", flush=True)
    print(f"  complete in {time.time()-t0:.0f}s", flush=True)
    return rows, summary


if __name__ == "__main__":
    run_experiment(sampler="wolff")
