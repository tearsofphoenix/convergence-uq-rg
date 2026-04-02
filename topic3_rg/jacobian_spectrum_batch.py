"""
Batch Jacobian spectrum analysis with multi-seed / multi-sample aggregation.

输出:
  outputs/rg_bench_wolff/jacobian_batch/raw.json
  outputs/rg_bench_wolff/jacobian_batch/summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from topic3_rg.jacobian_spectrum import train_model, compute_jacobian_spectrum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Jacobian spectra across seeds and samples.")
    parser.add_argument("--models", nargs="+", default=["Linear", "MLP", "CNN"])
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.4407)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eq-steps", type=int, default=1000)
    parser.add_argument("--sampler", choices=["metropolis", "wolff"], default="wolff")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--out-dir", default="outputs/rg_bench_wolff/jacobian_batch")
    return parser.parse_args()


def summarize_group(records: list[dict]) -> dict:
    spectral = np.array([r["spectral_norm"] for r in records], dtype=float)
    frob = np.array([r["frobenius_norm"] for r in records], dtype=float)
    rank = np.array([r["rank_eps_1e-6"] for r in records], dtype=float)
    top_vals = np.array([r["top_singular_values"][:5] for r in records], dtype=float)
    return {
        "n_records": len(records),
        "spectral_norm_mean": float(np.mean(spectral)),
        "spectral_norm_std": float(np.std(spectral, ddof=1)) if len(records) > 1 else 0.0,
        "frobenius_norm_mean": float(np.mean(frob)),
        "frobenius_norm_std": float(np.std(frob, ddof=1)) if len(records) > 1 else 0.0,
        "rank_mean": float(np.mean(rank)),
        "rank_std": float(np.std(rank, ddof=1)) if len(records) > 1 else 0.0,
        "top5_singular_mean": np.mean(top_vals, axis=0).tolist(),
        "top5_singular_std": np.std(top_vals, axis=0, ddof=1).tolist() if len(records) > 1 else [0.0] * top_vals.shape[1],
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_records: list[dict] = []
    summary: dict[str, dict] = {}

    print("=" * 70, flush=True)
    print("  BATCH JACOBIAN SPECTRUM ANALYSIS", flush=True)
    print(f"  models={args.models} L={args.L} beta={args.beta:.4f} sampler={args.sampler}", flush=True)
    print(f"  seeds={args.seeds} sample_count={args.sample_count}", flush=True)
    print("=" * 70, flush=True)

    for model_name in args.models:
        group_records: list[dict] = []
        for seed in args.seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            model, train_mse, test_mse, fine_test, coarse_test = train_model(
                model_name=model_name,
                L=args.L,
                beta=args.beta,
                seed=seed,
                n_train=args.n_train,
                n_test=args.n_test,
                epochs=args.epochs,
                batch_size=args.batch_size,
                eq_steps=args.eq_steps,
                sampler=args.sampler,
            )
            for sample_index in range(min(args.sample_count, len(fine_test))):
                record = compute_jacobian_spectrum(model, fine_test[sample_index])
                record.update({
                    "model": model_name,
                    "L": args.L,
                    "beta": args.beta,
                    "seed": seed,
                    "sample_index": sample_index,
                    "sampler": args.sampler,
                    "train_mse": train_mse,
                    "test_mse": test_mse,
                    "target_norm": float(np.linalg.norm(coarse_test[sample_index])),
                })
                raw_records.append(record)
                group_records.append(record)
        summary[model_name] = summarize_group(group_records)
        summary[model_name]["train_mse_mean"] = float(np.mean([r["train_mse"] for r in group_records]))
        summary[model_name]["test_mse_mean"] = float(np.mean([r["test_mse"] for r in group_records]))
        print(
            f"  {model_name}: spectral_norm_mean={summary[model_name]['spectral_norm_mean']:.4f}, "
            f"top5_mean={summary[model_name]['top5_singular_mean'][:3]}",
            flush=True,
        )

    raw_path = out_dir / "raw.json"
    with open(raw_path, "w") as f:
        json.dump(raw_records, f, indent=2)

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "config": {
                "models": args.models,
                "L": args.L,
                "beta": args.beta,
                "seeds": args.seeds,
                "n_train": args.n_train,
                "n_test": args.n_test,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "eq_steps": args.eq_steps,
                "sampler": args.sampler,
                "sample_count": args.sample_count,
            },
            "summary": summary,
        }, f, indent=2)

    print(f"  saved={raw_path}", flush=True)
    print(f"  saved={summary_path}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
