# Paper 3 Reproduction Notes

Date: 2026-04-02  
Author: Codex

## Scope

This document covers the current `paper3_rg.tex` workflow in its
negative-result / benchmark-design form.

The paper is organized around three evidence tiers:

1. Primary evidence: Wolff-sampled same-scale Ising benchmark
2. Methodological clarification: exploratory whole-lattice transfer
3. Diagnostic appendices: mixing, XY nonlinear coarse-graining, Jacobian spectra

## Main Outputs

### Primary Ising benchmark

- Directory: `outputs/rg_bench_wolff/cross_scale/`
- Key files:
  - `cross_scale_raw.csv`
  - `cross_scale_statistics.json`
  - `figures/model_comparison_bars.png`
  - `figures/temperature_dependence.png`
  - `figures/cross_scale_degradation.png`

Run:

```bash
RG_CROSS_SCALE_OUT_DIR=outputs/rg_bench_wolff/cross_scale \
conda run --no-capture-output -n hybridqml311 python topic3_rg/cross_scale_experiment.py
```

### Mixing diagnostics

- Directory: `outputs/rg_bench_wolff/mixing_analysis/`
- Key files:
  - `raw.csv`
  - `summary.json`

Run:

```bash
RG_MIXING_OUT_DIR=outputs/rg_bench_wolff/mixing_analysis \
conda run --no-capture-output -n hybridqml311 python - <<'PY'
from topic3_rg.mixing_analysis import run_mixing_analysis
run_mixing_analysis(
    L_values=[8, 16],
    betas=[0.30, 0.4407, 0.60],
    n_seeds=3,
    eq_steps=1000,
    n_sweeps=3000,
    max_lag=200,
    sampler="wolff",
)
PY
```

### Exploratory whole-lattice transfer

- Directory: `outputs/rg_bench_wolff/whole_lattice_transfer/`
- Key files:
  - `raw.csv`
  - `summary.json`

Run:

```bash
RG_WHOLE_LATTICE_OUT_DIR=outputs/rg_bench_wolff/whole_lattice_transfer \
conda run --no-capture-output -n hybridqml311 python - <<'PY'
from topic3_rg.whole_lattice_transfer import run_experiment
run_experiment(
    n_seeds=3,
    n_train=200,
    n_test=100,
    epochs=40,
    batch_size=32,
    sampler="wolff",
)
PY
```

## Diagnostic Branches

### XY nonlinear coarse-graining

- Single-scale main run: `outputs/xy_rg_main/`
- Multi-scale run: `outputs/xy_rg_multiscale/`

Recommended multi-scale run:

```bash
XY_RG_OUT_DIR=outputs/xy_rg_multiscale \
conda run --no-capture-output -n hybridqml311 python topic3_rg/xy_rg_benchmark.py \
  --L-values 8 16 \
  --betas 0.60 1.12 1.50 \
  --models Linear MLP CNN \
  --n-seeds 10 \
  --n-train 200 \
  --n-test 100 \
  --epochs 40 \
  --batch-size 32
```

Pairwise statistics:

```bash
conda run --no-capture-output -n hybridqml311 python topic3_rg/xy_rg_analysis.py \
  --raw outputs/xy_rg_multiscale/xy_rg_raw.csv \
  --out outputs/xy_rg_multiscale/xy_rg_stats.json
```

### Jacobian summaries

- Single-run Jacobian files: `outputs/rg_bench_wolff/jacobian/`
- Batch summaries:
  - `outputs/rg_bench_wolff/jacobian_batch/summary.json`
  - `outputs/rg_bench_wolff/jacobian_batch_L16/summary.json`

Batch run example:

```bash
conda run --no-capture-output -n hybridqml311 python -m topic3_rg.jacobian_spectrum_batch \
  --models Linear MLP CNN \
  --L 8 \
  --beta 0.4407 \
  --seeds 42 43 44 \
  --n-train 200 \
  --n-test 40 \
  --epochs 20 \
  --batch-size 16 \
  --sampler wolff \
  --sample-count 8 \
  --out-dir outputs/rg_bench_wolff/jacobian_batch
```

## Paper Build

```bash
python scripts/build_changed_tex.py --files paper3_rg.tex
```

Expected artifacts:

- `paper3_rg.pdf`
- no undefined references/citations in `paper3_rg.log`

## Current Positioning

The present paper should be read as:

- a benchmark-design negative-result paper on local Ising block-spin learning
- with exploratory transfer results used as methodological clarification
- and XY / Jacobian branches used as diagnostics, not as co-equal positive claims
