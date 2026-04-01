"""
Sampling autocorrelation and mixing diagnostics for the Ising experiments.

输出:
  outputs/rg_bench/mixing_analysis/summary.json
  outputs/rg_bench/mixing_analysis/raw.csv
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from topic3_rg.ising import IsingConfig, IsingModel


OUT_DIR = Path("outputs/rg_bench/mixing_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    var = np.var(x)
    if var <= 1e-12:
        return np.ones(max_lag + 1, dtype=np.float64)
    n = len(x)
    acf = np.empty(max_lag + 1, dtype=np.float64)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = np.dot(x[:-lag], x[lag:]) / ((n - lag) * var)
    return acf


def integrated_autocorrelation_time(acf: np.ndarray) -> float:
    tau = 0.5
    for lag in range(1, len(acf)):
        if acf[lag] <= 0:
            break
        tau += acf[lag]
    return max(tau, 0.5)


def effective_sample_size(n: int, tau_int: float) -> float:
    return float(n / (2.0 * tau_int)) if tau_int > 0 else float(n)


def run_mixing_analysis(L_values: list[int] | None = None, betas: list[float] | None = None,
                        n_seeds: int = 3, eq_steps: int = 1000, n_sweeps: int = 4000,
                        max_lag: int = 200) -> tuple[list[dict], dict]:
    if L_values is None:
        L_values = [8, 16]
    if betas is None:
        betas = [0.30, 0.4407, 0.60]

    rows: list[dict] = []
    summary: dict[str, dict] = {}
    seeds = list(range(42, 42 + n_seeds))
    t0 = time.time()

    print("\n" + "=" * 70, flush=True)
    print("  MIXING / AUTOCORRELATION ANALYSIS", flush=True)
    print(f"  Output   : {OUT_DIR}", flush=True)
    print("=" * 70, flush=True)

    for L in L_values:
        for beta in betas:
            energy_taus = []
            mag_taus = []
            acc_means = []
            for seed in seeds:
                np.random.seed(seed)
                model = IsingModel(IsingConfig(L=L, beta=beta, h=0.0, J=1.0))
                series = model.time_series(n_sweeps=n_sweeps, eq_steps=eq_steps)

                energy_acf = autocorrelation(series["energy"], max_lag=max_lag)
                mag_acf = autocorrelation(series["magnetization_abs"], max_lag=max_lag)
                tau_e = integrated_autocorrelation_time(energy_acf)
                tau_m = integrated_autocorrelation_time(mag_acf)
                ess_e = effective_sample_size(n_sweeps, tau_e)
                ess_m = effective_sample_size(n_sweeps, tau_m)
                acc = float(np.mean(series["acceptance"]))

                rows.append({
                    "L": L,
                    "beta": beta,
                    "seed": seed,
                    "tau_energy": tau_e,
                    "tau_magnetization_abs": tau_m,
                    "ess_energy": ess_e,
                    "ess_magnetization_abs": ess_m,
                    "acceptance_mean": acc,
                    "energy_mean": float(np.mean(series["energy"])),
                    "magnetization_abs_mean": float(np.mean(series["magnetization_abs"])),
                })

                energy_taus.append(tau_e)
                mag_taus.append(tau_m)
                acc_means.append(acc)

            summary[f"L{L}_beta_{beta:.4f}"] = {
                "L": L,
                "beta": beta,
                "n_seeds": n_seeds,
                "eq_steps": eq_steps,
                "n_sweeps": n_sweeps,
                "tau_energy_mean": float(np.mean(energy_taus)),
                "tau_energy_std": float(np.std(energy_taus, ddof=1)) if len(energy_taus) > 1 else 0.0,
                "tau_magnetization_abs_mean": float(np.mean(mag_taus)),
                "tau_magnetization_abs_std": float(np.std(mag_taus, ddof=1)) if len(mag_taus) > 1 else 0.0,
                "acceptance_mean": float(np.mean(acc_means)),
                "acceptance_std": float(np.std(acc_means, ddof=1)) if len(acc_means) > 1 else 0.0,
            }
            print(
                f"  L={L} beta={beta:.4f}: "
                f"tau_E={np.mean(energy_taus):.1f}, "
                f"tau_|M|={np.mean(mag_taus):.1f}, "
                f"acc={np.mean(acc_means):.3f}",
                flush=True,
            )

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
    run_mixing_analysis()
