"""Benchmark — M4 Pro 实测性能"""
import sys, time
sys.path.insert(0, '/Users/isaac/clawd/research/hermes')

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
mx.set_default_device(mx.gpu)

def count_params(tree, acc=0):
    if isinstance(tree, mx.array): return acc + int(tree.size)
    if isinstance(tree, dict): return sum(count_params(v, acc) for v in tree.values())
    if isinstance(tree, (list, tuple)): return sum(count_params(x, acc) for x in tree)
    return acc

print("=" * 60)
print("M4 Pro 实测性能基准")
print(f"GPU: {mx.default_device()}")
print("=" * 60)

t0 = time.time()

# ═══ TOPIC 1: Neural Operators ══════════════════════════════
print("\n【课题一：神经算子 — DeepONet + FNO】")
print("-" * 45)

from topic1_convergence.deeponet import DeepONet
from topic1_convergence.fno import FNO1d, FNO2d

# DeepONet forward
t1 = time.time()
m_do = DeepONet(num_sensors=50, coord_dim=2, p=20)
u = mx.random.normal((8, 50)); y = mx.random.normal((8, 20, 2))
_ = m_do(u, y)
t_do = time.time() - t1
n_do = count_params(m_do.trainable_parameters())
print(f"  DeepONet forward  (8×50→8×20):  {t_do*1000:.1f}ms | 参数: {n_do:,}")

# DeepONet training step
opt_do = opt.Adam(learning_rate=1e-3)
params_do = dict(m_do.trainable_parameters())
steps = []
for i in range(20):
    ts = time.time()
    grads = mx.grad(lambda _: mx.mean(m_do(u, y)**2))(params_do)
    opt_do.update(params_do, grads)
    mx.eval(list(params_do.values()))
    steps.append(time.time() - ts)
print(f"  DeepONet train step (batch=8):  {sum(steps)/len(steps)*1000:.1f}ms/step")

# ConvergenceAnalyzer
t_a = time.time()
from topic1_convergence.analysis import ConvergenceAnalyzer
import numpy as np
analyzer = ConvergenceAnalyzer()
for pde in ["poisson_2d", "heat_1d", "burgers_1d"]:
    for solver in ["deeponet", "fno"]:
        N = [100, 500, 1000, 2000]
        errs = [10*n**-0.4 + 0.01*np.random.randn() for n in N]
        analyzer.fit_power_law(N, errs)
print(f"  ConvergenceAnalyzer (3×2×4=24 fits):  {(time.time()-t_a)*1000:.0f}ms")

# FNO1d
t_f1 = time.time()
for bs in [4, 8, 16]:
    m = FNO1d(num_sensors=128, width=32, modes=12, num_layers=4)
    x = mx.random.normal((bs, 128)); _ = m(x)
print(f"  FNO1d forward (batch 4/8/16, grid=128):  {(time.time()-t_f1)*1000:.1f}ms")

# FNO2d
t_f2 = time.time()
for gs in [16, 32, 64]:
    m = FNO2d(H=gs, width=32, modes=8, num_layers=3)
    x = mx.random.normal((4, gs, gs)); _ = m(x)
print(f"  FNO2d forward (batch=4, grid 16/32/64):  {(time.time()-t_f2)*1000:.1f}ms")

# ═══ TOPIC 2: UQ Calibration ══════════════════════════════════
print("\n【课题二：UQ校准 — MC Dropout / Conformal】")
print("-" * 45)
from topic2_uq.calibration import MCDropoutUQ, ConformalPredictor

class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 1)
    def __call__(self, x): return self.fc(x)

for npasses in [10, 20, 50]:
    mc = MCDropoutUQ(Dummy(), num_passes=npasses)
    X = mx.random.normal((32, 100))
    ts = time.time()
    mean_p, std_p, _ = mc.predict(X)
    mx.eval([mean_p, std_p])
    print(f"  MC Dropout ({npasses:2d} passes, 32×100):  {(time.time()-ts)*1000:.1f}ms")

cp = ConformalPredictor(alpha=0.1)
err_cal = mx.abs(mx.random.normal((500,)))
ts = time.time(); q = cp.calibrate(err_cal); mx.eval(q)
print(f"  Conformal calibrate (N=500):  {(time.time()-ts)*1000:.1f}ms")

X_test = mx.random.normal((200,))
ts = time.time(); lower, upper = cp.predict(X_test); mx.eval([lower, upper])
print(f"  Conformal predict (N=200):  {(time.time()-ts)*1000:.1f}ms")

# ═══ TOPIC 3: RG × NN / Ising / Turbulence ══════════════════
print("\n【课题三：RG×神经网络 — Ising / NN-as-RG / LBM】")
print("-" * 45)
from topic3_rg.ising import IsingModel, IsingConfig, BlockSpinRG
from topic3_rg.neural_rg import NNAsRGBlock
from topic3_rg.turbulence import LBM2D, verify_kolmogorov_spectrum, inertial_range_analysis

for L in [8, 16, 32]:
    ts = time.time()
    ising = IsingModel(IsingConfig(L=L, beta=0.44))
    ising.initialize(); ising.equilibriate(200)
    _, obs = ising.sample(100, stride=5)
    print(f"  Ising MC (L={L:2d}, 100 samples):  {(time.time()-ts)*1000:.0f}ms  M={obs.magnetization:.4f}")

for L in [16, 32, 64]:
    rg = BlockSpinRG(block_size=2)
    ising = IsingModel(IsingConfig(L=L, beta=0.44))
    ising.initialize(); ising.equilibriate(200)
    ts = time.time()
    for _ in range(50): rg.block_spin_transform(ising.state)
    print(f"  Block spin RG (L={L:2d}, 50 transforms):  {(time.time()-ts)*1000:.0f}ms")

for L in [8, 16, 32]:
    nnrg = NNAsRGBlock(L=L)
    x = 2.0*mx.random.randint(0, 2, (4, L, L)) - 1.0
    ts = time.time()
    for _ in range(20): _ = nnrg(x)
    mx.eval([])
    print(f"  NN-as-RG (L={L:2d}, 4×{L}×{L}, 20 iter):  {(time.time()-ts)*1000:.0f}ms")

for nx in [32, 64, 128]:
    lbm = LBM2D(nx=nx, ny=nx, reynolds=200.0, u0=0.05)
    for _ in range(50): lbm.step()
    ts = time.time()
    for _ in range(100): lbm.step()
    elapsed = time.time() - ts
    print(f"  LBM ({nx:3d}×{nx}, Re=200, 100 steps):  {elapsed*1000:.0f}ms  ({100/elapsed:.0f} steps/s)")

lbm = LBM2D(nx=64, ny=64, reynolds=200.0)
for _ in range(300): lbm.step()
ts = time.time()
k, Ek = verify_kolmogorov_spectrum((lbm.ux, lbm.uy), dx=1.0)
slope = inertial_range_analysis(k, Ek, k_start=3, k_end=15)
print(f"  Kolmogorov谱 (64×64 grid):  {(time.time()-ts)*1000:.0f}ms  slope={slope:.3f}")

print(f"\n{'='*60}")
print(f"全部完成 | 总耗时: {(time.time()-t0):.1f}s")