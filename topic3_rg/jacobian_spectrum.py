"""
Jacobian spectrum analysis for Paper 3 RG models.

目标:
  - 在给定 (L, beta) 与模型类型上训练一个 block-spin 预测器
  - 对单个测试样本求输出关于输入的 Jacobian
  - 导出奇异值谱与 J J^T 的特征值谱，作为后续临界指数分析的起点

输出:
  outputs/rg_bench/jacobian/*.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from topic3_rg.ising import BlockSpinRG, IsingConfig, IsingModel
from topic3_rg.cross_scale_experiment import FlatMLP, LinearModel, CNNBlockSpin, RGInformedMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a block-spin model and compute Jacobian spectrum.")
    parser.add_argument("--model", choices=["MLP", "Linear", "CNN", "RGMLP"], default="CNN")
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.4407)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eq-steps", type=int, default=1000)
    parser.add_argument("--sampler", choices=["metropolis", "wolff"], default="wolff")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        default="outputs/rg_bench_wolff/jacobian",
        help="Directory for JSON outputs.",
    )
    return parser.parse_args()


def generate_dataset(L: int, beta: float, n_samples: int, seed: int,
                     eq_steps: int, sampler: str) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    rg = BlockSpinRG(block_size=2)
    ising = IsingModel(IsingConfig(L=L, beta=beta, h=0.0, J=1.0, sampler=sampler))
    ising.equilibriate(eq_steps)

    fine_samples = []
    coarse_samples = []
    for _ in range(n_samples + 100):
        ising.sampling_step(ising.state)
        fine = ising.state.copy().astype(np.float32)
        coarse = rg.block_spin_transform(fine).astype(np.float32)
        fine_samples.append(fine.reshape(-1))
        coarse_samples.append(coarse.reshape(-1))
    return np.array(fine_samples[:n_samples]), np.array(coarse_samples[:n_samples])


def build_model(model_name: str, L: int) -> nn.Module:
    L_in = L * L
    L_out = (L // 2) * (L // 2)
    if model_name == "MLP":
        return FlatMLP(L_in=L_in, L_out=L_out)
    if model_name == "Linear":
        return LinearModel(L_in=L_in, L_out=L_out)
    if model_name == "CNN":
        return CNNBlockSpin(L=L)
    if model_name == "RGMLP":
        return RGInformedMLP(L=L)
    raise ValueError(model_name)


def train_model(model_name: str, L: int, beta: float, seed: int, n_train: int,
                n_test: int, epochs: int, batch_size: int, eq_steps: int,
                sampler: str) -> tuple[nn.Module, float, float, np.ndarray, np.ndarray]:
    fine_train, coarse_train = generate_dataset(L, beta, n_train, seed, eq_steps, sampler)
    fine_test, coarse_test = generate_dataset(L, beta, n_test, seed + 10000, eq_steps, sampler)

    model = build_model(model_name, L)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(fine_train), torch.from_numpy(coarse_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    for _ in range(epochs):
        model.train()
        for x_b, y_b in loader:
            optimizer.zero_grad()
            pred = model(x_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_mse = criterion(model(torch.from_numpy(fine_train)), torch.from_numpy(coarse_train)).item()
        test_mse = criterion(model(torch.from_numpy(fine_test)), torch.from_numpy(coarse_test)).item()
    return model, float(train_mse), float(test_mse), fine_test, coarse_test


def compute_jacobian_spectrum(model: nn.Module, sample: np.ndarray) -> dict:
    x0 = torch.tensor(sample, dtype=torch.float32, requires_grad=True)

    def forward_fn(x_flat: torch.Tensor) -> torch.Tensor:
        return model(x_flat.unsqueeze(0)).squeeze(0)

    jacobian = torch.autograd.functional.jacobian(forward_fn, x0, vectorize=True)
    jacobian = jacobian.detach()
    singular_values = torch.linalg.svdvals(jacobian)
    gram = jacobian @ jacobian.T
    gram_eigenvalues = torch.linalg.eigvalsh(gram)

    singular_sorted = torch.sort(singular_values, descending=True).values.cpu().numpy()
    gram_sorted = torch.sort(gram_eigenvalues, descending=True).values.cpu().numpy()

    return {
        "jacobian_shape": list(jacobian.shape),
        "frobenius_norm": float(torch.linalg.norm(jacobian).item()),
        "spectral_norm": float(singular_sorted[0]) if singular_sorted.size else 0.0,
        "rank_eps_1e-6": int(np.sum(singular_sorted > 1e-6)),
        "top_singular_values": singular_sorted[:10].tolist(),
        "top_gram_eigenvalues": gram_sorted[:10].tolist(),
        "all_singular_values": singular_sorted.tolist(),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, train_mse, test_mse, fine_test, coarse_test = train_model(
        model_name=args.model,
        L=args.L,
        beta=args.beta,
        seed=args.seed,
        n_train=args.n_train,
        n_test=args.n_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eq_steps=args.eq_steps,
        sampler=args.sampler,
    )
    sample_index = max(0, min(args.sample_index, len(fine_test) - 1))
    spectrum = compute_jacobian_spectrum(model, fine_test[sample_index])

    record = {
        "model": args.model,
        "L": args.L,
        "beta": args.beta,
        "seed": args.seed,
        "sampler": args.sampler,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "sample_index": sample_index,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "target_norm": float(np.linalg.norm(coarse_test[sample_index])),
        **spectrum,
    }

    out_path = out_dir / f"{args.model}_L{args.L}_beta{args.beta:.4f}_seed{args.seed}_{args.sampler}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    print("=" * 70, flush=True)
    print("  JACOBIAN SPECTRUM ANALYSIS", flush=True)
    print(f"  model={args.model} L={args.L} beta={args.beta:.4f} sampler={args.sampler}", flush=True)
    print(f"  train_mse={train_mse:.6f} test_mse={test_mse:.6f}", flush=True)
    print(f"  top singular values={record['top_singular_values'][:5]}", flush=True)
    print(f"  saved={out_path}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
