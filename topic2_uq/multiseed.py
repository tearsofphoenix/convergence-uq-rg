"""
Multi-seed validation for UQ benchmark.
Runs key configs (Conformal + Deep Ensemble, Poisson2D/Heat1D/NavierStokes2D, nominal=0.90)
across 3 seeds to get mean ± std.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from topic2_uq.benchmark import UQExperiment

PDES = ["poisson_2d", "heat_1d", "navier_stokes_2d"]
TARGET_METHODS = ["Conformal", "Deep Ensemble"]   # match exact strings from benchmark.py
SEEDS = [42, 2024, 7777]
NOMINAL = 0.90
PRECISION = "fp32"

all_results = {}

for pde in PDES:
    print(f"\n{'='*60}")
    print(f"  PDE: {pde}")
    print(f"{'='*60}")
    all_results[pde] = {}
    for method in TARGET_METHODS:
        all_results[pde][method] = {"seeds": {}, "summary": {}}
        for seed in SEEDS:
            exp = UQExperiment(
                pde=pde,
                precisions=[PRECISION],
                coverage_levels=[NOMINAL],
                seed=seed,
            )
            results = exp.run()   # list of UQResult
            # Filter to the method + precision + nominal we want
            for r in results:
                if r.method == method and r.precision == PRECISION and r.nominal == NOMINAL:
                    all_results[pde][method]["seeds"][seed] = {
                        "coverage": r.empirical_coverage,
                        "ece": r.ece,
                        "width": r.avg_width,
                        "n_cal": r.n_cal,
                        "n_test": r.n_test,
                    }
                    print(f"  {method:14s} seed={seed} "
                          f"cov={r.empirical_coverage:.3f} ece={r.ece:.3f} w={r.avg_width:.4f}")
                    break

        # Compute summary statistics
        seeds_data = all_results[pde][method]["seeds"]
        covs = [v["coverage"] for v in seeds_data.values()]
        eces = [v["ece"] for v in seeds_data.values()]
        wids = [v["width"] for v in seeds_data.values()]
        mean = lambda x: sum(x)/len(x)
        std = lambda x: (sum((v - mean(x))**2 for v in x) / len(x))**0.5
        all_results[pde][method]["summary"] = {
            "nominal": NOMINAL,
            "precision": PRECISION,
            "seeds": list(seeds_data.keys()),
            "coverage_mean": mean(covs),
            "coverage_std": std(covs),
            "ece_mean": mean(eces),
            "ece_std": std(eces),
            "width_mean": mean(wids),
            "width_std": std(wids),
        }
        print(f"  >> {method:14s} SUMMARY: "
              f"cov={mean(covs):.3f}±{std(covs):.3f}  "
              f"ece={mean(eces):.3f}±{std(eces):.3f}  "
              f"w={mean(wids):.4f}±{std(wids):.4f}")

out_path = Path("/Users/isaacliu/workspace/convergence-uq-rg/outputs/uq_bench/multiseed_results.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
