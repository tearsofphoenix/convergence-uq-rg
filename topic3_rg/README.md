# Topic 3: Renormalization Group × Neural Networks

**物理核心**: Wilson RG形式主义与神经网络的深度交叉

## 核心问题
- NN是否在隐式实现近似RG变换？
- RG理论能否解释神经网络的跨尺度推广性？
- RG启发的神经闭合模型能否在湍流惯性范围保持-5/3幂律？

## 代码结构
```
topic3_rg/
├── ising.py             # Ising模型 + MC + 块自旋RG
├── neural_rg.py         # NN-as-RG-block + 尺度传递实验
├── turbulence.py        # Navier-Stokes LBM + 神经闭合模型
├── literature/          # RG + ML参考文献
└── results/             # 实验结果 + PhyRG-Bench
```

## 运行
```bash
# Ising临界指数提取
cd /Users/isaac/clawd/research/topic3_rg
python3 ising.py

# RG-NN连接实验
python3 neural_rg.py

# 湍流模拟
python3 turbulence.py
```

## 关键实验
1. **RG Flow Recovery**: NN能否学习正确的块自旋变换方向？
2. **Critical Exponent Extraction**: 从NN权重中提取临界指数
3. **Scale Transferability**: 跨尺度推广性测试
4. **Fixed Point Detection**: NN是否收敛到RG不动点？

## 环境
- Python: `hybridqml311` conda环境（numpy/scipy用于Ising MC）
- MLX: 神经RG block训练
- OpenFOAM: 湍流CFD（Mac Mini 2018）
