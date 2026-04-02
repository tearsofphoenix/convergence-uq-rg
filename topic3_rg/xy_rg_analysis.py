"""
Pairwise statistical analysis for XY nonlinear coarse-graining benchmark.

输入:
  outputs/xy_rg_main/xy_rg_raw.csv

输出:
  outputs/xy_rg_main/xy_rg_stats.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze XY benchmark raw results.")
    parser.add_argument("--raw", default="outputs/xy_rg_main/xy_rg_raw.csv")
    parser.add_argument("--out", default="outputs/xy_rg_main/xy_rg_stats.json")
    return parser.parse_args()


def cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    n1, n2 = len(g1), len(g2)
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / max(n1 + n2 - 2, 1)
    return float((np.mean(g2) - np.mean(g1)) / np.sqrt(pooled + 1e-12))


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    results: dict[str, dict] = {}

    for L in sorted(df["L"].unique()):
        sub_L = df[df["L"] == L].copy()
        for beta in sorted(sub_L["beta"].unique()):
            sub = sub_L[sub_L["beta"] == beta].copy()
            key = f"L{int(L)}_beta_{beta:.4f}"
            results[key] = {}
            for metric in ["test_mse", "test_mean_abs_angle_error"]:
                results[key][metric] = {}
                for lhs, rhs in [("Linear", "MLP"), ("Linear", "CNN"), ("MLP", "CNN")]:
                    g1 = sub[sub["model"] == lhs][metric].to_numpy(dtype=float)
                    g2 = sub[sub["model"] == rhs][metric].to_numpy(dtype=float)
                    welch_t, welch_p = stats.ttest_ind(g1, g2, equal_var=False)
                    mw_u, mw_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                    results[key][metric][f"{lhs}_vs_{rhs}"] = {
                        "L": int(L),
                        "beta": float(beta),
                        lhs: {
                            "mean": float(np.mean(g1)),
                            "std": float(np.std(g1, ddof=1)),
                            "n": int(len(g1)),
                        },
                        rhs: {
                            "mean": float(np.mean(g2)),
                            "std": float(np.std(g2, ddof=1)),
                            "n": int(len(g2)),
                        },
                        "welch_t": float(welch_t),
                        "welch_p": float(welch_p),
                        "mannwhitney_u": float(mw_u),
                        "mannwhitney_p": float(mw_p),
                        "cohens_d_rhs_minus_lhs": cohens_d(g1, g2),
                    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(out_path)


if __name__ == "__main__":
    main()
