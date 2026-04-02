# Neural Operator Learning: Convergence, UQ, and Renormalization Group

**Author:** Xeon Pitar
**Hardware:** MacBook Pro M4 (Apple Silicon, 48GB) + Mac Mini 2018 (Intel x86\_64, 64GB)
**Framework:** MLX (Apple Silicon) + FEniCS / OpenFOAM (CPU baselines)

---

Three standalone papers on neural operator learning for partial differential equations.

### Paper 1 — Convergence Rate Theory
`paper1_convergence.tex` | Target: FoCM / SINUM / NeurIPS Theory

Minimax convergence rates for DeepONet and FNO. Proves $N^{-\beta/(d+\beta)}$
optimal rates, spectral $M^{-\delta}$ for FNO, and dimensional efficiency
bounds breaking the curse of dimensionality in low-rank PDEs. MLX experiments
on 5 PDE types confirm all predictions.

### Paper 2 — UQ Calibration Benchmark
`paper2_uq.tex` | Target: JCP / Scientific Reports

PDE-Bench-UQ: first systematic UQ benchmark for neural PDE solvers.
Three UQ methods (MC Dropout, Deep Ensembles, Conformal Prediction) across
5 PDE types $\times$ 3 architectures $\times$ 3 precision levels (FP32/INT8/INT4).
Conformal Prediction is most robust; INT4 quantization severely degrades
Bayesian methods but CP remains stable.

### Paper 3 — RG × Neural Networks
`paper3_rg.tex` | Target: PRE / JFM / NeurIPS

Benchmark-design negative result on RG-like learning claims. The current Paper 3
version centers a Wolff-sampled 2D Ising majority-vote benchmark showing that,
on this local block-spin task, a linear baseline beats a standard MLP at
criticality. Whole-lattice transfer, XY nonlinear coarse-graining, and Jacobian
spectra are retained as diagnostics rather than co-equal primary evidence.

---

## Repository Structure

```
convergence-uq-rg/
├── paper1_convergence.tex   # Topic 1: Convergence theory
├── paper2_uq.tex            # Topic 2: UQ calibration
├── paper3_rg.tex            # Topic 3: RG × NN
├── shared/                  # Shared code (PDE suite, hardware utils, data manager)
│   ├── pde_suite.py         # Canonical PDE test suite
│   ├── hardware.py           # Hardware detection & config
│   └── data_manager.py      # Experiment data management
├── topic1_convergence/      # DeepONet + FNO MLX implementations
│   ├── deeponet.py          # DeepONet (MLX)
│   ├── fno.py               # Fourier Neural Operator (MLX)
│   └── analysis.py          # Convergence analysis & power-law fitting
├── topic2_uq/               # UQ calibration experiments
│   └── calibration.py       # MC Dropout, Deep Ensembles, Conformal Prediction
├── topic3_rg/               # RG × Neural Network experiments
│   ├── ising.py             # 2D Ising model (Metropolis MC, exact diag)
│   ├── neural_rg.py         # NNAsRGBlock, scale transfer, fixed-point detection
│   └── turbulence.py        # D2Q9 LBM, neural closure for Navier-Stokes
├── benchmark.py             # Full benchmark runner (6.7s on M4 Pro)
├── environment.yml          # Conda environment spec
└── setup.sh                # Environment setup script
```

## Setup

```bash
# Clone
git clone https://github.com/xeonpitar/convergence-uq-rg.git
cd convergence-uq-rg

# Conda environment (MacBook Pro M4)
conda env create -f environment.yml
conda activate hybridqml311

# Run full benchmark (MLX, Apple Silicon)
python benchmark.py
```

## Build Papers

```bash
pdflatex paper1_convergence.tex
pdflatex paper2_uq.tex
pdflatex paper3_rg.tex
```

## Auto-Build PDFs

仓库现在带有 `.githooks/` 本地 hook：

- `pre-commit`: 编译已 stage 的独立 `.tex` 文件，并自动 stage 对应 `.pdf`
- `post-checkout` / `post-merge` / `post-commit`: 对 git 变更涉及的独立 `.tex` 重新生成 `.pdf`

手工触发也可以：

```bash
python scripts/build_changed_tex.py --files paper3_rg.tex
python scripts/build_changed_tex.py --staged --stage-pdfs
```

## Paper 3 Repro

See [docs/paper3_rg_repro.md](/Users/isaac/workspace/AI/hermes/convergence-uq-rg/docs/paper3_rg_repro.md) for the current negative-result / benchmark-design workflow, including:

- Wolff-sampled main benchmark outputs
- Exploratory transfer and mixing diagnostics
- XY nonlinear diagnostic branch
- Jacobian batch summaries

Additional submission-oriented notes:

- [docs/paper3_contribution_summary.md](/Users/isaac/workspace/AI/hermes/convergence-uq-rg/docs/paper3_contribution_summary.md)
- [docs/paper3_submission_checklist.md](/Users/isaac/workspace/AI/hermes/convergence-uq-rg/docs/paper3_submission_checklist.md)
- [docs/paper3_cover_letter.md](/Users/isaac/workspace/AI/hermes/convergence-uq-rg/docs/paper3_cover_letter.md)

##发表路线图

| Timeline | Venue | Topic |
|----------|-------|-------|
| Year 1 Q3 | FoCM / SINUM / NeurIPS Theory | Paper 1: Convergence theory |
| Year 2 Q1 | JCP / Scientific Reports | Paper 2: PDE-Bench-UQ |
| Year 2 Q3 | PRE / JFM / NeurIPS | Paper 3: RG × Neural Networks |
| Year 3 Q1 | NeurIPS | Paper 4: RG turbulence application |
| Year 3 Q3 | Thesis | --- |

## License

TBD
