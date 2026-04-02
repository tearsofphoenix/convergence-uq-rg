"""
Exploratory nonlinear coarse-graining benchmark on the 2D XY model.

设计目标:
  - 构造一个比 Ising majority-vote 更接近“真正非线性 coarse-graining”的监督任务
  - 输入使用每个格点的 (cos θ, sin θ) 两通道表示
  - 目标使用 2x2 block 的 circular mean（归一化向量），其中归一化步骤是关键非线性
  - 比较 Linear / MLP / CNN 在该任务上的 same-scale 表现

输出:
  outputs/xy_rg_pilot/xy_rg_raw.csv
  outputs/xy_rg_pilot/xy_rg_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


Array = np.ndarray
OUT_DIR = Path(os.environ.get("XY_RG_OUT_DIR", "outputs/xy_rg_main"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cpu"


@dataclass
class XYConfig:
    L: int
    beta: float
    J: float = 1.0
    proposal_width: float = 1.2


class XYModel:
    """2D XY model with local Metropolis updates on angle variables."""

    def __init__(self, config: XYConfig):
        self.config = config
        self.L = config.L
        self.beta = config.beta
        self.J = config.J
        self.proposal_width = config.proposal_width
        self.state: Array | None = None

    def initialize(self) -> Array:
        self.state = np.random.uniform(-math.pi, math.pi, size=(self.L, self.L)).astype(np.float32)
        return self.state.copy()

    def neighbors(self, i: int, j: int) -> tuple[float, float, float, float]:
        L = self.L
        s = self.state
        return (
            float(s[(i - 1) % L, j]),
            float(s[(i + 1) % L, j]),
            float(s[i, (j - 1) % L]),
            float(s[i, (j + 1) % L]),
        )

    def delta_energy(self, i: int, j: int, new_theta: float) -> float:
        old_theta = float(self.state[i, j])
        neighbor_angles = self.neighbors(i, j)
        old_term = sum(math.cos(old_theta - theta_n) for theta_n in neighbor_angles)
        new_term = sum(math.cos(new_theta - theta_n) for theta_n in neighbor_angles)
        return -self.J * (new_term - old_term)

    def metropolis_sweep(self) -> float:
        accepted = 0
        total = self.L * self.L
        for _ in range(total):
            i = np.random.randint(self.L)
            j = np.random.randint(self.L)
            old_theta = float(self.state[i, j])
            proposal = old_theta + np.random.uniform(-self.proposal_width, self.proposal_width)
            new_theta = ((proposal + math.pi) % (2 * math.pi)) - math.pi
            dE = self.delta_energy(i, j, new_theta)
            if dE <= 0.0 or np.random.rand() < math.exp(-self.beta * dE):
                self.state[i, j] = new_theta
                accepted += 1
        return accepted / total

    def equilibrate(self, n_sweeps: int = 1000) -> Array:
        if self.state is None:
            self.initialize()
        for _ in range(n_sweeps):
            self.metropolis_sweep()
        return self.state.copy()


def angles_to_features(theta: Array) -> Array:
    """将角变量映射到 (cos θ, sin θ) 两通道。"""
    return np.stack([np.cos(theta), np.sin(theta)], axis=0).astype(np.float32)


def circular_block_mean(theta: Array, block_size: int = 2, eps: float = 1e-8) -> Array:
    """对每个 block 取 circular mean，并输出归一化后的二维向量。"""
    L = theta.shape[0]
    out_L = L // block_size
    coarse = np.zeros((2, out_L, out_L), dtype=np.float32)
    for bi in range(out_L):
        for bj in range(out_L):
            block = theta[
                bi * block_size:(bi + 1) * block_size,
                bj * block_size:(bj + 1) * block_size,
            ]
            x = float(np.sum(np.cos(block)))
            y = float(np.sum(np.sin(block)))
            norm = math.sqrt(x * x + y * y)
            coarse[0, bi, bj] = x / max(norm, eps)
            coarse[1, bi, bj] = y / max(norm, eps)
    return coarse


def generate_xy_dataset(L: int, beta: float, n_samples: int, seed: int,
                        eq_sweeps: int = 1000, stride: int = 10) -> tuple[Array, Array]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = XYModel(XYConfig(L=L, beta=beta))
    model.equilibrate(eq_sweeps)

    fine, coarse = [], []
    for step in range(n_samples * stride):
        model.metropolis_sweep()
        if step % stride != 0:
            continue
        theta = model.state.copy()
        fine.append(angles_to_features(theta))
        coarse.append(circular_block_mean(theta))
    return np.array(fine[:n_samples], dtype=np.float32), np.array(coarse[:n_samples], dtype=np.float32)


class XYDataset(Dataset):
    def __init__(self, fine: Array, coarse: Array):
        self.fine = torch.from_numpy(fine)
        self.coarse = torch.from_numpy(coarse.reshape(len(coarse), -1))

    def __len__(self) -> int:
        return len(self.fine)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fine[idx], self.coarse[idx]


class LinearXY(nn.Module):
    def __init__(self, L: int):
        super().__init__()
        self.linear = nn.Linear(2 * L * L, 2 * (L // 2) * (L // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.reshape(x.shape[0], -1))


class MLPXY(nn.Module):
    def __init__(self, L: int, hidden: int = 256):
        super().__init__()
        dim_in = 2 * L * L
        dim_out = 2 * (L // 2) * (L // 2)
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1))


class CNNXY(nn.Module):
    def __init__(self, L: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 2, kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).reshape(x.shape[0], -1)


def build_model(model_name: str, L: int) -> nn.Module:
    if model_name == "Linear":
        return LinearXY(L)
    if model_name == "MLP":
        return MLPXY(L)
    if model_name == "CNN":
        return CNNXY(L)
    raise ValueError(model_name)


def angular_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_xy = pred.reshape(pred.shape[0], 2, -1)
    target_xy = target.reshape(target.shape[0], 2, -1)
    pred_angles = torch.atan2(pred_xy[:, 1], pred_xy[:, 0])
    target_angles = torch.atan2(target_xy[:, 1], target_xy[:, 0])
    diff = torch.atan2(torch.sin(pred_angles - target_angles), torch.cos(pred_angles - target_angles))
    return float(torch.mean(torch.abs(diff)).item())


def train_and_eval(model_name: str, L: int, beta: float, seed: int,
                   n_train: int, n_test: int, epochs: int, batch_size: int) -> dict:
    fine_train, coarse_train = generate_xy_dataset(L=L, beta=beta, n_samples=n_train, seed=seed)
    fine_test, coarse_test = generate_xy_dataset(L=L, beta=beta, n_samples=n_test, seed=seed + 10_000)

    model = build_model(model_name, L).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    loader = DataLoader(XYDataset(fine_train, coarse_train), batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for x_b, y_b in loader:
            x_b = x_b.to(DEVICE)
            y_b = y_b.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        x_train = torch.from_numpy(fine_train).to(DEVICE)
        y_train = torch.from_numpy(coarse_train.reshape(len(coarse_train), -1)).to(DEVICE)
        x_test = torch.from_numpy(fine_test).to(DEVICE)
        y_test = torch.from_numpy(coarse_test.reshape(len(coarse_test), -1)).to(DEVICE)
        pred_train = model(x_train)
        pred_test = model(x_test)
        train_mse = float(criterion(pred_train, y_train).item())
        test_mse = float(criterion(pred_test, y_test).item())
        test_ang = angular_error(pred_test, y_test)

    return {
        "model": model_name,
        "L": L,
        "beta": beta,
        "seed": seed,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "test_mean_abs_angle_error": test_ang,
    }


def summarize(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    by_key: dict[tuple, list[dict]] = {}
    for row in rows:
        by_key.setdefault((row["L"], row["beta"], row["model"]), []).append(row)

    for (L, beta, model), group in by_key.items():
        test_mse = np.array([g["test_mse"] for g in group], dtype=float)
        test_ang = np.array([g["test_mean_abs_angle_error"] for g in group], dtype=float)
        summary[f"L{L}_beta_{beta:.4f}_{model}"] = {
            "L": L,
            "beta": beta,
            "model": model,
            "mean_test_mse": float(np.mean(test_mse)),
            "std_test_mse": float(np.std(test_mse, ddof=1)) if len(test_mse) > 1 else 0.0,
            "mean_abs_angle_error": float(np.mean(test_ang)),
            "std_abs_angle_error": float(np.std(test_ang, ddof=1)) if len(test_ang) > 1 else 0.0,
            "n": len(group),
        }
    return summary


def run_experiment(
    L_values: list[int] | None = None,
    betas: list[float] | None = None,
    models: list[str] | None = None,
    n_seeds: int = 3,
    n_train: int = 200,
    n_test: int = 100,
    epochs: int = 40,
    batch_size: int = 32,
) -> tuple[list[dict], dict]:
    if L_values is None:
        L_values = [8]
    if betas is None:
        betas = [0.60, 1.12, 1.50]
    if models is None:
        models = ["Linear", "MLP", "CNN"]

    print("\n" + "=" * 70, flush=True)
    print("  XY NONLINEAR COARSE-GRAINING PILOT", flush=True)
    print("  Target   : 2x2 block circular mean (normalized vector)", flush=True)
    print(f"  Output   : {OUT_DIR}", flush=True)
    print("=" * 70, flush=True)

    rows: list[dict] = []
    total = len(L_values) * len(betas) * len(models) * n_seeds
    idx = 0
    t0 = time.time()

    for L in L_values:
        for beta in betas:
            for model_name in models:
                for seed in range(42, 42 + n_seeds):
                    idx += 1
                    if idx == 1 or idx % 5 == 0:
                        elapsed = time.time() - t0
                        rate = idx / elapsed if elapsed > 0 else 0.0
                        eta = (total - idx) / rate if rate > 0 else 0.0
                        print(
                            f"  run {idx}/{total}: model={model_name} L={L} beta={beta:.2f} "
                            f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                            flush=True,
                        )
                    rows.append(
                        train_and_eval(
                            model_name=model_name,
                            L=L,
                            beta=beta,
                            seed=seed,
                            n_train=n_train,
                            n_test=n_test,
                            epochs=epochs,
                            batch_size=batch_size,
                        )
                    )

    summary = summarize(rows)

    raw_path = OUT_DIR / "xy_rg_raw.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = OUT_DIR / "xy_rg_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  saved: {raw_path}", flush=True)
    print(f"  saved: {summary_path}", flush=True)
    print(f"  complete in {time.time() - t0:.0f}s", flush=True)
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run nonlinear coarse-graining benchmark on the 2D XY model.")
    parser.add_argument("--L-values", type=int, nargs="+", default=[8])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.60, 1.12, 1.50])
    parser.add_argument("--models", nargs="+", default=["Linear", "MLP", "CNN"])
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        L_values=args.L_values,
        betas=args.betas,
        models=args.models,
        n_seeds=args.n_seeds,
        n_train=args.n_train,
        n_test=args.n_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
