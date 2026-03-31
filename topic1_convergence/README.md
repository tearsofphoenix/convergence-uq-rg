# Topic 1: Neural Operator Convergence Theory

**理论核心**: DeepONet / FNO / PINNs 的收敛率数学理论

## 核心问题
- 神经算子何时优于传统FEM？收敛率是多少？
- 能否建立极小极大的最优性证明？
- PAC-Bayes能否为神经算子提供非平凡推广界？

## 代码结构
```
topic1_convergence/
├── deeponet.py          # DeepONet MLX实现 + 收敛率实验
├── fno.py               # FNO (Fourier Neural Operator) MLX实现
├── analysis.py          # 收敛率分析工具 + 理论对比
├── literature/          # 参考文献库
└── results/             # 实验结果存储
```

## 运行
```bash
# DeepONet 收敛率实验
cd topic1_convergence
PYTHONPATH=/Users/isaac/clawd/research python3 -c "
from deeponet import convergence_experiment
convergence_experiment('poisson_2d')
"
```

## 环境
- Python: `hybridqml311` conda环境
- MLX: Apple Silicon GPU加速
- 备选: FEniCS (Mac Mini 2018) 作为传统方法基准
