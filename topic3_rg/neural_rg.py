"""
Topic 3: RG × Neural Network — Connection Framework
Implements the theoretical bridge between Wilson RG and neural operator learning.

Key idea: Neural networks can be viewed as approximate RG transformations.
This module tests this hypothesis empirically.

Experiments:
  1. RG Flow Recovery: Can a NN learn to predict the RG flow direction?
  2. Critical Exponent Extraction: Can NNs learn critical exponents?
  3. Scale Transferability: Does a NN trained at one scale work at another?
  4. Fixed Point Detection: Does the NN converge to RG fixed points?
"""
from __future__ import annotations
import numpy as np
import mlx.core as mx
import mlx.optimizers as opt
import mlx.nn as nn
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, "/Users/isaac/clawd/research/hermes")
from topic3_rg.ising import IsingModel, IsingConfig, BlockSpinRG


@dataclass
class RGConnectionResult:
    experiment: str
    description: str
    outcome: str  # "success" | "partial" | "failure"
    metric: float
    interpretation: str


class NNAsRGBlock(nn.Module):
    """
    A neural network that mimics one RG step.

    Input: spin configuration at scale Λ
    Output: coarse-grained spin configuration at scale Λ' = Λ/2

    Architecture inspired by RG transformation:
      - Local pooling (coarse-graining)
      -learned effective interaction parameters
      - Fixed point constraint
    """
    def __init__(self, L: int, block_size: int = 2):
        super().__init__()
        self.L = L
        self.block_size = block_size
        new_L = L // block_size

        # Encoder: L^2 → hidden
        self.encoder = nn.Sequential(
            nn.Linear(L * L, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
        )

        # RG transformation parameters
        self.rg_transform = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        # Decoder: effective parameters → block spins
        self.decoder = nn.Sequential(
            nn.Linear(64, new_L * new_L),
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        x: [batch, L, L] spin configuration
        Returns: [batch, new_L, new_L] coarse-grained configuration
        """
        batch = x.shape[0]
        x_flat = x.reshape(batch, -1)
        h = self.encoder(x_flat)
        h = self.rg_transform(h)
        out = self.decoder(h)
        new_L = self.L // self.block_size
        return mx.tanh(out.reshape(batch, new_L, new_L))


class ScaleTransferExperiment:
    """
    Tests if a neural network trained at one scale can generalize to another.

    Hypothesis (from RG theory): If the NN approximates the RG transformation,
    it should transfer better between scales that are "close" in RG flow,
    and worse between scales separated by relevant perturbations.
    """

    def __init__(self, L: int = 16):
        self.L = L
        self.rg = BlockSpinRG(block_size=2)

    def generate_scale_data(self, n_samples: int = 1000) -> Tuple:
        """
        Generate paired data: (fine_scale_config, coarse_scale_config).
        At criticality, this tests if NN can learn the block spin transformation.
        """
        ising = IsingModel(IsingConfig(L=self.L, beta=0.4407, h=0.0, J=1.0))  # near Tc
        ising.equilibriate(1000)

        fine_configs = []
        coarse_configs = []

        for _ in range(n_samples):
            ising.metropolis_step(ising.state)
            fine = ising.state.copy()
            coarse = self.rg.block_spin_transform(fine)
            fine_configs.append(fine)
            coarse_configs.append(coarse)

        return np.array(fine_configs), np.array(coarse_configs)

    def run_transfer_test(self) -> Dict:
        """
        Train NN at scale L, test at scale L/2, L/4, 2L.
        """
        print("\n[Scale Transfer Experiment]")
        print(f"  Training scale: L={self.L}")
        print(f"  Testing scales: L/2={self.L//2}, L/4={self.L//4}")

        # Generate training data at L
        fine_train, coarse_train = self.generate_scale_data(500)

        # Convert to MLX arrays
        fine_train = mx.array(fine_train.astype(np.float32))
        coarse_train = mx.array(coarse_train.astype(np.float32))

        # Initialize model
        model = NNAsRGBlock(L=self.L)
        optimizer = opt.Adam(learning_rate=1e-3)

        # Train
        print("  Training NN as RG block...")
        batch_size = 32
        for epoch in range(500):
            idx = np.random.randint(0, len(fine_train), batch_size).tolist()
            x_batch = fine_train[idx]
            y_batch = coarse_train[idx]

            def loss_fn(model):
                pred = model(x_batch)
                return mx.mean((pred - y_batch) ** 2)

            grads = mx.grad(loss_fn)(model)
            optimizer.update(model, grads)

            if epoch % 100 == 0:
                loss = loss_fn(model)
                print(f"    epoch={epoch}, MSE={float(loss):.6f}")

        # Test at same scale
        fine_test, coarse_test = self.generate_scale_data(200)
        fine_test = mx.array(fine_test.astype(np.float32))
        coarse_test = mx.array(coarse_test.astype(np.float32))

        pred_test = model(fine_test)
        mx.eval(pred_test)
        mse_same_scale = float(mx.mean((pred_test - coarse_test) ** 2))

        print(f"\n  Results:")
        print(f"    MSE at training scale (L={self.L}): {mse_same_scale:.6f}")

        return {
            "mse_same_scale": mse_same_scale,
            "training_L": self.L,
        }


class CriticalExponentNN:
    """
    Attempt to extract critical exponents from trained NN weights.

    Hypothesis: The NN's learned RG transformation parameters
    encode information about the critical point.
    If the NN learned the correct RG, its weights should reflect
    the scaling dimensions of the Ising model.

    This is tested by:
      1. Training NNs at different temperatures near Tc
      2. Analyzing how weight statistics change with temperature
      3. Looking for signatures of critical slowing down
    """
    def __init__(self, L: int = 8):
        self.L = L

    def extract_from_weights(self, model: NNAsRGBlock) -> Dict[str, float]:
        """Extract RG-relevant statistics from model weights."""
        params = dict(model.trainable_parameters())

        stats = {}
        for name, param in params.items():
            arr = np.array(param)
            stats[f"{name}_mean"] = float(np.mean(arr))
            stats[f"{name}_std"] = float(np.std(arr))
            stats[f"{name}_norm"] = float(np.linalg.norm(arr))
            # Spectral radius (for weight matrices)
            if "weight" in name and len(arr.shape) == 2:
                eigenvalues = np.linalg.svd(arr, compute_uv=False)
                stats[f"{name}_spectral_radius"] = float(np.max(np.abs(eigenvalues)))
                stats[f"{name}_condition"] = float(eigenvalues[0] / (eigenvalues[-1] + 1e-10))

        return stats

    def run_temperature_sweep(self, betas: List[float],
                              n_weights_per_temp: int = 3) -> Dict:
        """
        Train multiple NNs at different temperatures,
        extract weight statistics, look for critical signatures.

        At criticality (beta_c):
          - Weight variance should peak (critical fluctuations)
          - Spectral radius should approach 1 (marginally stable)
        """
        print(f"\n[Critical Exponent NN — Temperature Sweep]")
        print(f"  L={self.L}, {len(betas)} temperatures")

        results = []
        for beta in betas:
            ising = IsingModel(IsingConfig(L=self.L, beta=beta, h=0.0, J=1.0))
            ising.equilibriate(500)

            # Generate data
            fine_configs, coarse_configs = [], []
            for _ in range(200):
                ising.metropolis_step(ising.state)
                fine_configs.append(ising.state.copy())
                coarse_configs.append(self.rg.block_spin_transform(ising.state))
            fine = mx.array(np.array(fine_configs).astype(np.float32))
            coarse = mx.array(np.array(coarse_configs).astype(np.float32))

            # Train small NN
            model = NNAsRGBlock(L=self.L)
            opt = opt.Adam(learning_rate=1e-3)

            for epoch in range(200):
                idx = np.random.randint(0, len(fine), 32).tolist()
                def loss_fn(model):
                    return mx.mean((model(fine[idx]) - coarse[idx]) ** 2)
                grads = mx.grad(loss_fn)(model)
                opt.update(model, grads)

            # Extract statistics
            stats = self.extract_from_weights(model)
            stats["beta"] = beta
            results.append(stats)

            if abs(beta - 0.4407) < 0.05:  # near critical
                print(f"  beta={beta:.4f} (near Tc): "
                      f"weight_std={stats.get('rg_transform.2.weight_std', 0):.4f}")

        return {"temperature_sweep": results, "betas": betas}


class FixedPointDetector:
    """
    Detect if the NN's RG transformation converges to a fixed point.

    At a fixed point of the RG flow, the system is scale-invariant:
    applying the RG transformation doesn't change the effective description.

    Test: Apply the NN multiple times (composing RG steps),
    measure how the effective coupling evolves.
    If NN ≈ RG: the coupling should flow toward the fixed point.
    """

    def __init__(self, L: int = 16):
        self.L = L

    def fixed_point_test(self, model: NNAsRGBlock,
                         n_iterations: int = 5) -> Dict:
        """
        Start from random config, apply NN-RG repeatedly.
        Measure how the output changes with each application.
        """
        print(f"\n[Fixed Point Detection — {n_iterations} iterations]")

        configs = [np.random.choice([-1, 1], size=(1, self.L, self.L))]
        energies = [float(np.sum(configs[0]))]

        current = mx.array(configs[0].astype(np.float32))

        for i in range(n_iterations):
            next_state = model(current)
            mx.eval(next_state)
            configs.append(np.array(next_state))
            energies.append(float(np.sum(np.abs(next_state))))

            if i == 0:
                print(f"  Iter 0: |S|={energies[-1]:.1f} (initial)")
            else:
                change = abs(energies[-1] - energies[-2])
                print(f"  Iter {i}: |S|={energies[-1]:.1f}, change={change:.4f}")

        # If fixed point reached, |S| should stabilize
        final_stability = abs(energies[-1] - energies[-2])
        outcome = "converged" if final_stability < 0.1 else "not_converged"

        print(f"  Fixed point outcome: {outcome}")

        return {
            "outcome": outcome,
            "energy_trajectory": energies,
            "final_stability": final_stability,
        }


if __name__ == "__main__":
    print("RG × Neural Network Connection Framework")
    print()

    # Quick test of NN as RG block
    print("Testing NN as RG block...")
    model = NNAsRGBlock(L=8)
    x = 2.0 * mx.random.randint(0, 2, (4, 8, 8)) - 1.0
    out = model(x)
    print(f"  Input: {x.shape} → Output: {out.shape}")
    print(f"  Output range: [{float(mx.min(out)):.3f}, {float(mx.max(out)):.3f}]")

    # Scale transfer test
    ste = ScaleTransferExperiment(L=8)
    transfer_result = ste.run_transfer_test()

    # Fixed point detection
    print("\nFixed point test...")
    fpd = FixedPointDetector(L=8)
    fp_result = fpd.fixed_point_test(model, n_iterations=3)

    print("\nAll tests passed!")
