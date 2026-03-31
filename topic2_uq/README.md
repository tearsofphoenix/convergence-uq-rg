# Topic 2: UQ Calibration for Neural PDE Solvers

**方法论核心**: 不确定性量化（MC Dropout / Deep Ensemble / Conformal Prediction）的校准性测量

## 核心问题
- 90%预测区间是否真的包含90%真实值？
- 4-bit量化对UQ可靠性的影响？
- Apple Silicon上神经求解器的可信度基准？

## 代码结构
```
topic2_uq/
├── calibration.py       # UQ校准工具 + 基准测试套件
├── conformal.py         # Conformal Prediction实现
├── results/              # 实验结果存储
└── notebooks/            # 分析notebooks
```

## 运行
```bash
cd /Users/isaac/clawd/research/topic2_uq
PYTHONPATH=/Users/isaac/clawd/research python3 -c "
from calibration import run_uq_benchmark
results = run_uq_benchmark('poisson_2d')
"
```

## 关键指标
- Coverage: 预测区间覆盖率（nominal vs empirical）
- ECE: 期望校准误差
- NLL: 负对数似然
- Width: 平均区间宽度

## 环境
- Python: `hybridqml311` conda环境
- MLX: MC Dropout, Deep Ensemble推理
- FEniCS: PDE基准求解（Mac Mini）
