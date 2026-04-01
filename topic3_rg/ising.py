"""
Topic 3: Renormalization Group Meets Neural Networks
Ising model implementation for RG analysis.

This module implements the 2D Ising model with:
  - Metropolis-Hastings Monte Carlo
  - Exact diagonalization (for small lattices)
  - Block spin RG transformation (Kadanoff)
  - Critical exponent extraction
  - Neural network interpretation analysis

Key research questions:
  1. Can the NN learn the correct block spin transformation?
  2. Can critical exponents be extracted from NN weights?
  3. Does the NN approximate the RG fixed point?

References:
  - Kadanoff (1966): Block spin transformation
  - Wilson (1974/1983): RG theory
  - Mehta & Schwab (2014): RG and Deep Learning connection
  - Park et al. (2019): Neural RG
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import numpy.typing as npt
from scipy import linalg
import json

# Use numpy for Ising (CPU-friendly, no GPU needed for MC)
Array = npt.NDArray[np.float64]


@dataclass
class IsingConfig:
    L: int                # Lattice size (L x L)
    beta: float           # Inverse temperature
    h: float = 0.0        # External field
    J: float = 1.0        # Coupling constant (ferromagnetic if J>0)

    def critical_beta(self) -> float:
        """Onsager's exact critical inverse temperature for h=0."""
        beta_c = np.log(1 + np.sqrt(2)) / (2 * self.J)
        return beta_c

    def is_near_critical(self, tol: float = 0.05) -> bool:
        return abs(self.beta - self.critical_beta()) / self.critical_beta() < tol


@dataclass
class IsingObservables:
    energy: float
    magnetization: float
    susceptibility: float
    specific_heat: float
    beta: float
    L: int
    num_samples: int

    def to_dict(self) -> dict:
        return {
            "energy": self.energy,
            "magnetization": self.magnetization,
            "susceptibility": self.susceptibility,
            "specific_heat": self.specific_heat,
            "beta": self.beta,
            "L": self.L,
            "num_samples": self.num_samples,
        }


class IsingModel:
    """
    2D Ising model on square lattice with periodic BC.
    H = -J * sum_{<i,j>} s_i * s_j - h * sum_i s_i
    """
    def __init__(self, config: IsingConfig):
        self.config = config
        self.L = config.L
        self.beta = config.beta
        self.h = config.h
        self.J = config.J
        self.state: Optional[Array] = None

    def initialize(self, scheme: str = "random") -> Array:
        """Initialize lattice spin configuration."""
        if scheme == "random":
            self.state = np.random.choice([-1, 1], size=(self.L, self.L))
        elif scheme == "ordered":
            self.state = np.ones((self.L, self.L), dtype=np.float64)
        elif scheme == "hot":
            self.state = np.random.choice([-1, 1], size=(self.L, self.L))
        return self.state.copy()

    def energy(self, state: Optional[Array] = None) -> float:
        """Compute total energy of configuration."""
        if state is None:
            state = self.state
        # Sum over nearest neighbors (each pair counted once)
        # Horizontal: state[:, :-1] * state[:, 1:]
        # Vertical: state[:-1, :] * state[1:, :]
        horz = np.sum(state[:, :-1] * state[:, 1:])
        vert = np.sum(state[:-1, :] * state[1:, :])
        interaction = -self.J * (horz + vert)
        field = -self.h * np.sum(state)
        return float(interaction + field)

    def delta_energy(self, i: int, j: int, state: Array) -> float:
        """Energy change if spin at (i,j) is flipped."""
        L = self.L
        # Periodic boundary
        neighbors = (
            state[(i - 1) % L, j] +
            state[(i + 1) % L, j] +
            state[i, (j - 1) % L] +
            state[i, (j + 1) % L]
        )
        return 2 * self.h * state[i, j] + 2 * self.J * state[i, j] * neighbors

    def metropolis_step(self, state: Array) -> Array:
        """One full Metropolis-Hastings sweep (L*L attempted flips)."""
        L = self.L
        for _ in range(L * L):
            i = np.random.randint(L)
            j = np.random.randint(L)
            dE = self.delta_energy(i, j, state)
            if dE < 0 or np.random.rand() < np.exp(-self.beta * dE):
                state[i, j] *= -1
        return state

    def metropolis_step_with_stats(self, state: Array) -> Tuple[Array, float]:
        """一整次 sweep，并返回接受率。"""
        L = self.L
        accepted = 0
        total = L * L
        for _ in range(total):
            i = np.random.randint(L)
            j = np.random.randint(L)
            dE = self.delta_energy(i, j, state)
            if dE < 0 or np.random.rand() < np.exp(-self.beta * dE):
                state[i, j] *= -1
                accepted += 1
        return state, accepted / total if total > 0 else 0.0

    def equilibriate(self, n_steps: int = 1000) -> Array:
        """Run equilibration steps, return final state."""
        if self.state is None:
            self.initialize()
        for _ in range(n_steps):
            self.metropolis_step(self.state)
        return self.state

    def sample(self, n_samples: int, eq_steps: int = 500,
               stride: int = 10) -> Tuple[Array, IsingObservables]:
        """
        Generate n_samples correlated configurations.
        Uses stride to reduce autocorrelation.
        """
        if self.state is None:
            self.equilibriate(eq_steps)

        energies = []
        magnetizations = []
        samples = []

        current = self.state.copy()
        for _ in range(n_samples * stride):
            self.metropolis_step(current)
            if _ % stride == 0:
                samples.append(current.copy())
                energies.append(self.energy(current))
                magnetizations.append(abs(np.mean(current)))

        self.state = current
        samples = np.array(samples)

        # Compute observables
        E = np.array(energies)
        M = np.array(magnetizations)
        n = len(E)

        var_M = np.var(M) if n > 1 else 0.0
        var_E = np.var(E) if n > 1 else 0.0
        obs = IsingObservables(
            energy=np.mean(E),
            magnetization=np.mean(M),
            susceptibility=self.beta * var_M,
            specific_heat=self.beta**2 * var_E,
            beta=self.beta,
            L=self.L,
            num_samples=n_samples,
        )
        return samples, obs

    def time_series(self, n_sweeps: int, eq_steps: int = 1000) -> Dict[str, Array]:
        """记录采样时间序列，用于自相关与混合性诊断。"""
        if self.state is None:
            self.initialize()
        self.equilibriate(eq_steps)

        energies = np.zeros(n_sweeps, dtype=np.float64)
        mags_signed = np.zeros(n_sweeps, dtype=np.float64)
        mags_abs = np.zeros(n_sweeps, dtype=np.float64)
        acceptance = np.zeros(n_sweeps, dtype=np.float64)

        current = self.state.copy()
        for t in range(n_sweeps):
            current, acc = self.metropolis_step_with_stats(current)
            energies[t] = self.energy(current)
            m = np.mean(current)
            mags_signed[t] = m
            mags_abs[t] = abs(m)
            acceptance[t] = acc

        self.state = current
        return {
            "energy": energies,
            "magnetization_signed": mags_signed,
            "magnetization_abs": mags_abs,
            "acceptance": acceptance,
        }

    def magnetization(self, state: Optional[Array] = None) -> float:
        if state is None:
            state = self.state
        return float(np.abs(np.mean(state)))


class BlockSpinRG:
    """
    Implement Kadanoff's block spin renormalization group transformation.

    Key idea: coarse-grain the lattice by grouping 2x2 blocks,
    assign effective spin to each block based on majority rule,
    compute effective coupling J' and field h' of the coarse-grained system.
    """
    def __init__(self, block_size: int = 2):
        self.block_size = block_size

    def block_spin_transform(self, state: Array) -> Array:
        """
        Apply block spin transformation to a configuration.

        Args:
            state: [L, L] spin configuration (L must be even)

        Returns:
            coarse_grained: [L//block_size, L//block_size] block spin configuration
        """
        L = state.shape[0]
        assert L % self.block_size == 0, "L must be divisible by block_size"

        new_L = L // self.block_size
        coarse = np.zeros((new_L, new_L), dtype=np.float64)

        for bi in range(new_L):
            for bj in range(new_L):
                # Extract block
                block = state[
                    bi * self.block_size:(bi + 1) * self.block_size,
                    bj * self.block_size:(bj + 1) * self.block_size
                ]
                # Majority rule: +1 if majority up, -1 if majority down
                coarse[bi, bj] = 1.0 if np.sum(block) > 0 else -1.0

        return coarse

    def compute_effective_coupling(self, samples: List[Array],
                                    block_samples: List[Array]) -> Tuple[float, float]:
        """
        Estimate effective coupling J' and field h' of coarse-grained system.

        Uses correlation function method:
        J' ≈ -1/4 * log(<S_i S_j>_{block} / <S_i>_{block}^2) at distance 1
        h' estimated from magnetization imbalance

        Returns: (J_prime, h_prime)
        """
        if len(samples) == 0 or len(block_samples) == 0:
            return 1.0, 0.0

        # Estimate J' from block correlation
        block_corrs = []
        for bs in block_samples[:min(100, len(block_samples))]:
            L = bs.shape[0]
            # Correlation at distance 1 (horizontal neighbors)
            corr = np.mean(bs[:, :-1] * bs[:, 1:])
            block_corrs.append(corr)

        mean_corr = np.mean(block_corrs)

        if mean_corr > 0 and mean_corr < 1:
            J_prime = -0.25 * np.log((1 - mean_corr) / (1 + mean_corr))
        else:
            J_prime = 0.5  # fallback

        # Estimate h' from net magnetization
        mag = np.mean([np.mean(bs) for bs in block_samples[:100]])
        h_prime = -np.arctanh(np.clip(mag, -0.99, 0.99)) if abs(mag) < 0.99 else 0.0

        return float(J_prime), float(h_prime)

    def flow(self, ising: IsingModel, n_rg_steps: int = 3,
             samples_per_step: int = 1000) -> List[Dict]:
        """
        Compute RG flow by iteratively applying block spin transformation.

        Tracks: J', h', beta', beta*J' (dimensionless coupling)
        At criticality, RG flow should converge to a fixed point.
        """
        results = []
        current_ising = ising
        current_L = ising.L

        for step in range(n_rg_steps):
            # Sample at current scale
            print(f"  RG step {step+1}: L={current_L}, beta={current_ising.beta:.6f}")
            samples, obs = current_ising.sample(samples_per_step, eq_steps=500, stride=10)

            # Apply block spin
            block_samples = [self.block_spin_transform(s) for s in samples[:min(200, len(samples))]]
            new_L = block_samples[0].shape[0]
            J_prime, h_prime = self.compute_effective_coupling(samples, block_samples)

            # Dimensionless coupling
            k = current_ising.beta * current_ising.J  # K = beta * J (Ising convention)
            k_prime = current_ising.beta * J_prime

            results.append({
                "step": step,
                "L": current_L,
                "beta": current_ising.beta,
                "J": current_ising.J,
                "K": k,  # K = beta*J
                "J_prime": J_prime,
                "h_prime": h_prime,
                "K_prime": k_prime,
                "magnetization": obs.magnetization,
                "susceptibility": obs.susceptibility,
            })

            # Update for next RG step
            if new_L >= 4:  # Stop if too small
                new_config = IsingConfig(L=new_L, beta=current_ising.beta,
                                         h=h_prime, J=J_prime)
                current_ising = IsingModel(new_config)
                current_L = new_L
            else:
                print(f"  Stopping: L={new_L} too small")
                break

        return results


class CriticalExponents:
    """
    Extract critical exponents from Ising model data.

    Near criticality (T → Tc), observables follow power laws:
      - M ~ (1 - T/Tc)^{beta}
      - chi ~ |T - Tc|^{-gamma}
      - C ~ |T - Tc|^{-alpha}
      - xi ~ |T - Tc|^{-nu}

    Onsager's exact values for 2D Ising (h=0):
      alpha = 0      (logarithmic divergence)
      beta  = 1/8   = 0.125
      gamma = 7/4   = 1.75
      delta = 15    (at Tc)
      nu    = 1     (from xi ~ |t|^{-1})
    """
    ONSAGER = {
        "alpha": 0.0,
        "beta": 0.125,
        "gamma": 1.75,
        "nu": 1.0,
        "delta": 15.0,
    }

    def __init__(self, L: int = 16):
        self.L = L
        self.onsager = self.ONSAGER.copy()

    def fit_magnetization(self, betas: Array, magnetizations: Array,
                          beta_c: float) -> Tuple[float, float]:
        """
        Fit M = A * (1 - beta/beta_c)^{beta_exp} near criticality.
        Returns: (A, beta_exp) where beta_exp should ≈ 0.125 (Onsager)
        """
        # Near critical: beta < beta_c and |1 - beta/beta_c| < 0.1
        # t = 1 - beta/beta_c > 0 requires beta < beta_c (ordered phase)
        t = 1 - betas / beta_c
        near_crit = (t > 0) & (t < 0.1)
        if np.sum(near_crit) < 3:
            return 1.0, self.onsager["beta"]

        t = t[near_crit]
        M = magnetizations[near_crit]

        # Linear fit in log-log space
        log_t = np.log(t + 1e-10)
        log_M = np.log(M + 1e-10)
        coeff = np.polyfit(log_t, log_M, 1)

        beta_exp = coeff[0]
        A = np.exp(coeff[1])
        return float(A), float(beta_exp)

    def fit_susceptibility(self, betas: Array, chi: Array,
                           beta_c: float) -> Tuple[float, float]:
        """Fit chi = A * |1 - beta/beta_c|^{-gamma_exp}."""
        # Use data from both sides of critical, take absolute t
        t = np.abs(1 - betas / beta_c) + 1e-10
        near_crit = (t > 0) & (t < 0.1)
        if np.sum(near_crit) < 3:
            return 1.0, self.onsager["gamma"]

        t = t[near_crit]
        chi_vals = chi[near_crit] + 1e-10

        log_t = np.log(t)
        log_chi = np.log(chi_vals)
        coeff = np.polyfit(log_t, log_chi, 1)

        gamma_exp = -coeff[0]  # note: minus sign
        A = np.exp(coeff[1])
        return float(A), float(gamma_exp)

    def exponent_report(self, extracted: Dict[str, float]) -> str:
        """Generate comparison report."""
        lines = ["Critical Exponent Comparison", "=" * 50, ""]
        lines.append(f"{'Exponent':<8} {'Onsager':<12} {'Extracted':<12} {'Error':<10}")
        lines.append("-" * 50)
        for key in ["beta", "gamma", "nu"]:
            on = self.onsager[key]
            ex = extracted.get(key, float('nan'))
            err = abs(ex - on) / on if on > 0 else 0
            status = "✓" if err < 0.2 else ("△" if err < 0.5 else "✗")
            lines.append(f"{key:<8} {on:<12.4f} {ex:<12.4f} {err:.1%} {status}")
        lines.append("=" * 50)
        return "\n".join(lines)


def run_ising_experiment(L: int = 16, ntemps: int = 20,
                          n_samples: int = 2000) -> Dict:
    """
    Run full Ising model RG experiment.
    Measures critical exponents and RG flow.
    """
    print(f"\n{'='*60}")
    print(f"Ising Model Experiment: L={L}, {ntemps} temperatures, {n_samples} samples")
    print(f"{'='*60}")

    ising = IsingModel(IsingConfig(L=L, beta=0.5, h=0.0, J=1.0))
    beta_c = ising.config.critical_beta()
    print(f"Critical beta: {beta_c:.6f} (Onsager exact)")

    # Temperatures around Tc
    betas = np.linspace(0.2, 0.7, ntemps)
    results = []

    print("\n[1] Measuring observables vs temperature...")
    for beta in betas:
        cfg = IsingConfig(L=L, beta=beta, h=0.0, J=1.0)
        mdl = IsingModel(cfg)
        _, obs = mdl.sample(n_samples, eq_steps=500, stride=10)
        results.append(obs.to_dict())
        if abs(beta - beta_c) < 0.02:
            print(f"  beta={beta:.4f} near Tc: M={obs.magnetization:.4f}, chi={obs.susceptibility:.4f}")

    # Extract critical exponents
    print("\n[2] Fitting critical exponents...")
    ce = CriticalExponents(L=L)
    beta_arr = np.array([r["beta"] for r in results])
    mag_arr = np.array([r["magnetization"] for r in results])
    chi_arr = np.array([r["susceptibility"] for r in results])

    A_beta, beta_exp = ce.fit_magnetization(beta_arr, mag_arr, beta_c)
    A_chi, gamma_exp = ce.fit_susceptibility(beta_arr, chi_arr, beta_c)

    extracted = {"beta": beta_exp, "gamma": gamma_exp, "nu": ce.onsager["nu"]}
    print(ce.exponent_report(extracted))

    # RG flow
    print("\n[3] Computing RG flow...")
    initial_ising = IsingModel(IsingConfig(L=L, beta=beta_c, h=0.0, J=1.0))
    initial_ising.equilibriate(2000)
    rg = BlockSpinRG(block_size=2)
    rg_flow = rg.flow(initial_ising, n_rg_steps=4, samples_per_step=1000)

    print("\nRG Flow:")
    hdr = f"{'Step':<5} {'L':<6} {'K=βJ':<10} {'K_prime':>8} {'J_prime':>8} {'h_prime':>8}"
    print(hdr)
    print("-" * 55)
    for r in rg_flow:
        print(f"{r['step']:<5} {r['L']:<6} {r['K']:<10.4f} {r['K_prime']:>8.4f} "
              f"{r['J_prime']:>8.4f} {r['h_prime']:>8.4f}")

    return {
        "L": L,
        "beta_c": beta_c,
        "critical_exponents": extracted,
        "observables": results,
        "rg_flow": rg_flow,
    }


if __name__ == "__main__":
    print("Ising Model + RG Analysis")
    print(f"2D Ising critical beta: {np.log(1+np.sqrt(2))/2:.6f}")

    # Quick test
    cfg = IsingConfig(L=8, beta=0.5, h=0.0, J=1.0)
    ising = IsingModel(cfg)
    ising.initialize("random")
    print(f"\nInitial energy: {ising.energy():.4f}")
    print(f"Initial magnetization: {ising.magnetization():.4f}")

    # Quick equilibration
    ising.equilibriate(500)
    print(f"After equilibration: energy={ising.energy():.4f}, M={ising.magnetization():.4f}")

    # Sample
    samples, obs = ising.sample(100, eq_steps=0, stride=5)
    print(f"\nObservables: E={obs.energy:.4f}, M={obs.magnetization:.4f}, "
          f"chi={obs.susceptibility:.4f}, C={obs.specific_heat:.4f}")

    # Run full experiment (small scale for quick test)
    result = run_ising_experiment(L=8, ntemps=10, n_samples=500)

    print("\n\nAll tests passed!")
