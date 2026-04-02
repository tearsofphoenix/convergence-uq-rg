"""
Sampling autocorrelation and mixing diagnostics for the Ising experiments.

输出:
  outputs/rg_bench/mixing_analysis/summary.json
  outputs/rg_bench/mixing_analysis/raw.csv
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from topic3_rg.ising import IsingConfig, IsingModel


OUT_DIR = Path(os.environ.get("RG_MIXING_OUT_DIR", "outputs/rg_bench/mixing_analysis"))
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
                        max_lag: int = 200, sampler: str = "wolff") -> tuple[list[dict], dict]:
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
    print(f"  Sampler  : {sampler}", flush=True)
    print(f"  Output   : {OUT_DIR}", flush=True)
    print("=" * 70, flush=True)

    for L in L_values:
        for beta in betas:
            energy_taus = []
            mag_taus = []
            sampler_stat_means = []
            accepted_means = []
            cluster_size_means = []
            cluster_count_means = []
            sampler_stat_name = "accepted_fraction"
            for seed in seeds:
                np.random.seed(seed)
                model = IsingModel(IsingConfig(L=L, beta=beta, h=0.0, J=1.0, sampler=sampler))
                series = model.time_series(n_sweeps=n_sweeps, eq_steps=eq_steps)

                energy_acf = autocorrelation(series["energy"], max_lag=max_lag)
                mag_acf = autocorrelation(series["magnetization_abs"], max_lag=max_lag)
                tau_e = integrated_autocorrelation_time(energy_acf)
                tau_m = integrated_autocorrelation_time(mag_acf)
                ess_e = effective_sample_size(n_sweeps, tau_e)
                ess_m = effective_sample_size(n_sweeps, tau_m)
                sampler_stat_name = str(series["sampler_stat_name"])
                sampler_stat = float(np.nanmean(series["sampler_stat"]))
                accepted_fraction = float(np.nanmean(series["accepted_fraction"])) if np.any(~np.isnan(series["accepted_fraction"])) else None
                mean_cluster_size = float(np.nanmean(series["mean_cluster_size"])) if np.any(~np.isnan(series["mean_cluster_size"])) else None
                clusters_per_sweep = float(np.nanmean(series["clusters_per_sweep"])) if np.any(~np.isnan(series["clusters_per_sweep"])) else None

                rows.append({
                    "L": L,
                    "beta": beta,
                    "sampler": sampler,
                    "seed": seed,
                    "tau_energy": tau_e,
                    "tau_magnetization_abs": tau_m,
                    "ess_energy": ess_e,
                    "ess_magnetization_abs": ess_m,
                    "sampler_stat_name": sampler_stat_name,
                    "sampler_stat_mean": sampler_stat,
                    "acceptance_mean": accepted_fraction,
                    "accepted_fraction_mean": accepted_fraction,
                    "mean_cluster_size": mean_cluster_size,
                    "clusters_per_sweep": clusters_per_sweep,
                    "energy_mean": float(np.mean(series["energy"])),
                    "magnetization_abs_mean": float(np.mean(series["magnetization_abs"])),
                })

                energy_taus.append(tau_e)
                mag_taus.append(tau_m)
                sampler_stat_means.append(sampler_stat)
                if accepted_fraction is not None:
                    accepted_means.append(accepted_fraction)
                if mean_cluster_size is not None:
                    cluster_size_means.append(mean_cluster_size)
                if clusters_per_sweep is not None:
                    cluster_count_means.append(clusters_per_sweep)

            summary[f"L{L}_beta_{beta:.4f}"] = {
                "L": L,
                "beta": beta,
                "sampler": sampler,
                "n_seeds": n_seeds,
                "eq_steps": eq_steps,
                "n_sweeps": n_sweeps,
                "tau_energy_mean": float(np.mean(energy_taus)),
                "tau_energy_std": float(np.std(energy_taus, ddof=1)) if len(energy_taus) > 1 else 0.0,
                "tau_magnetization_abs_mean": float(np.mean(mag_taus)),
                "tau_magnetization_abs_std": float(np.std(mag_taus, ddof=1)) if len(mag_taus) > 1 else 0.0,
                "sampler_stat_name": sampler_stat_name,
                "sampler_stat_mean": float(np.mean(sampler_stat_means)),
                "sampler_stat_std": float(np.std(sampler_stat_means, ddof=1)) if len(sampler_stat_means) > 1 else 0.0,
                "acceptance_mean": float(np.mean(accepted_means)) if accepted_means else None,
                "acceptance_std": float(np.std(accepted_means, ddof=1)) if len(accepted_means) > 1 else 0.0 if accepted_means else None,
                "accepted_fraction_mean": float(np.mean(accepted_means)) if accepted_means else None,
                "accepted_fraction_std": float(np.std(accepted_means, ddof=1)) if len(accepted_means) > 1 else 0.0 if accepted_means else None,
                "mean_cluster_size_mean": float(np.mean(cluster_size_means)) if cluster_size_means else None,
                "mean_cluster_size_std": float(np.std(cluster_size_means, ddof=1)) if len(cluster_size_means) > 1 else 0.0 if cluster_size_means else None,
                "clusters_per_sweep_mean": float(np.mean(cluster_count_means)) if cluster_count_means else None,
                "clusters_per_sweep_std": float(np.std(cluster_count_means, ddof=1)) if len(cluster_count_means) > 1 else 0.0 if cluster_count_means else None,
            }
            print(
                f"  L={L} beta={beta:.4f}: "
                f"tau_E={np.mean(energy_taus):.1f}, "
                f"tau_|M|={np.mean(mag_taus):.1f}, "
                f"{sampler_stat_name}={np.mean(sampler_stat_means):.3f}",
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
    run_mixing_analysis(sampler="wolff")
