"""
Paper 1 convergence experiment with n=10 seeds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topic1_convergence import convergence_experiment as ce

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 7777, 2024]
N_VALUES = [50, 100, 200, 400]
TEST_N = 100
TEST_SEED = 9000
OUT_PATH = ce.REPO_ROOT / "outputs" / "convergence" / "paper1_n10_stats.json"


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def run_single_seed(
    pde_name: str,
    seed: int,
    make_dataset_fn,
    num_sensors: int,
    coord_dim: int,
    extra_args: dict,
) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    test_ds = make_dataset_fn(
        N=TEST_N,
        seed=TEST_SEED,
        num_sensors=num_sensors,
        num_points=64,
        **extra_args,
    )

    errors = {}
    for N in N_VALUES:
        train_ds = make_dataset_fn(
            N=N,
            seed=seed,
            num_sensors=num_sensors,
            num_points=64,
            **extra_args,
        )
        loader = DataLoader(
            ce.PDEDataset(train_ds["u_sensors"], train_ds["y_coords"], train_ds["u_query"]),
            batch_size=32,
            shuffle=True,
        )
        model = ce.DeepONet(num_sensors=num_sensors, coord_dim=coord_dim, p=50, hidden=128)
        ce.train_deeponet(model, loader, epochs=ce.EPOCHS)
        mse = ce.eval_deeponet(model, test_ds["u_sensors"], test_ds["y_coords"], test_ds["u_query"])
        errors[str(N)] = float(mse)

    gamma_hat, log_c = ce.fit_power_law(N_VALUES, [errors[str(N)] for N in N_VALUES])
    return {
        "seed": seed,
        "gamma_hat": float(gamma_hat),
        "C": float(np.exp(log_c)),
        "mse": errors,
    }


def summarise_results(pde_name: str, runs: list[dict]) -> dict:
    mse_stats = {}
    for N in N_VALUES:
        values = [run["mse"][str(N)] for run in runs]
        mse_stats[str(N)] = {
            "mean": float(np.mean(values)),
            "std": sample_std(values),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    gamma_values = [run["gamma_hat"] for run in runs]
    theory = ce.THEORY[pde_name]["alpha"] * 2.0
    return {
        "pde": pde_name,
        "theory_gamma": float(theory),
        "seeds": runs,
        "mse_stats": mse_stats,
        "gamma_mean": float(np.mean(gamma_values)),
        "gamma_std": sample_std(gamma_values),
        "gamma_min": float(np.min(gamma_values)),
        "gamma_max": float(np.max(gamma_values)),
        "ratio_mean_to_theory": float(np.mean(gamma_values) / theory),
        "mse_ratio_50_to_400": float(mse_stats["50"]["mean"] / mse_stats["400"]["mean"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pde",
        choices=["heat_1d", "poisson_2d", "all"],
        default="all",
    )
    args = parser.parse_args()

    experiments = []
    if args.pde in {"heat_1d", "all"}:
        experiments.append(
            (
                "heat_1d",
                ce.make_heat_dataset,
                32,
                1,
                {"t_final": 0.05},
            )
        )
    if args.pde in {"poisson_2d", "all"}:
        experiments.append(
            (
                "poisson_2d",
                ce.make_poisson_dataset,
                64,
                2,
                {"grid_size": 8, "n_modes": 3},
            )
        )

    output = {
        "python": str(Path(__file__).resolve()),
        "backend": str((ce.REPO_ROOT / "topic1_convergence" / "convergence_experiment.py").resolve()),
        "test_seed": TEST_SEED,
        "test_n": TEST_N,
        "n_values": N_VALUES,
        "seed_list": SEEDS,
        "epochs": ce.EPOCHS,
        "results": [],
    }

    for pde_name, make_dataset_fn, num_sensors, coord_dim, extra_args in experiments:
        print(f"\n=== {pde_name} ===", flush=True)
        runs = []
        for seed in SEEDS:
            print(f"seed={seed}", flush=True)
            run = run_single_seed(
                pde_name=pde_name,
                seed=seed,
                make_dataset_fn=make_dataset_fn,
                num_sensors=num_sensors,
                coord_dim=coord_dim,
                extra_args=extra_args,
            )
            runs.append(run)
            mse_text = ", ".join(f"N={N}:{run['mse'][str(N)]:.6e}" for N in N_VALUES)
            print(f"  gamma_hat={run['gamma_hat']:.6f} | {mse_text}", flush=True)

        summary = summarise_results(pde_name, runs)
        output["results"].append(summary)
        print(
            f"summary {pde_name}: gamma={summary['gamma_mean']:.6f} ± {summary['gamma_std']:.6f}; "
            + ", ".join(
                f"N={N}:{summary['mse_stats'][str(N)]['mean']:.6e} ± {summary['mse_stats'][str(N)]['std']:.6e}"
                for N in N_VALUES
            ),
            flush=True,
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nsaved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
