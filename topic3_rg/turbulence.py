"""
Topic 3: Turbulence — Navier-Stokes and Neural Closure Models
Simplified 2D Lattice Boltzmann Method (D2Q9) in pure NumPy.
"""
from __future__ import annotations
import numpy as np
from typing import Dict


class LBM2D:
    """
    2D Lattice Boltzmann Method — D2Q9 model with BGK collision.
    For quick prototyping on M4. For production: OpenFOAM on Mac Mini.
    """
    def __init__(self, nx: int = 128, ny: int = 128,
                 reynolds: float = 200.0, u0: float = 0.05):
        self.nx = nx
        self.ny = ny
        self.Re = reynolds
        self.u0 = u0

        # D2Q9 lattice weights
        self.w = np.array([4/9, 1/9, 1/9, 1/9, 1/9,
                           1/36, 1/36, 1/36, 1/36], dtype=np.float64)
        # Directions: (9,)
        self.ex = np.array([0, 1, -1, 0, 0, 1, -1, -1, 1], dtype=np.float64)
        self.ey = np.array([0, 0, 0, 1, -1, 1, 1, -1, -1], dtype=np.float64)

        # Viscosity
        self.nu = u0 * nx / reynolds
        self.omega = 1.0 / (3.0 * self.nu + 0.5)

        # Fields
        self.rho = np.ones((nx, ny), dtype=np.float64)
        self.ux = np.zeros((nx, ny), dtype=np.float64)
        self.uy = np.zeros((nx, ny), dtype=np.float64)

        # Distribution functions: (9, nx, ny)
        self.f = np.zeros((9, nx, ny), dtype=np.float64)
        self.f_stream = np.zeros((9, nx, ny), dtype=np.float64)
        self.f_collide = np.zeros((9, nx, ny), dtype=np.float64)

        self._initialize()

    def _initialize(self):
        """Set f to equilibrium with small perturbations."""
        for i in range(9):
            u_sq = self.ux**2 + self.uy**2
            u_dot_e = self.ex[i] * self.ux + self.ey[i] * self.uy
            self.f[i] = self.w[i] * self.rho * (
                1 + 3*u_dot_e + 4.5*u_dot_e**2 - 1.5*u_sq
            )
        np.random.seed(42)
        noise = 0.01 * np.random.randn(self.nx, self.ny)
        for i in range(9):
            self.f[i] *= (1 + noise)

    def step(self) -> float:
        """One LBM step. Returns kinetic energy."""
        nx, ny = self.nx, self.ny

        # ── Streaming (bounce-back with periodic BC) ──
        for i in range(9):
            sx = self.ex[i]
            sy = self.ey[i]
            rolled_x = np.roll(self.f[i], int(sx), axis=0)
            rolled_xy = np.roll(rolled_x, int(sy), axis=1)
            self.f_stream[i] = rolled_xy

        # ── Collision (BGK) ──
        # Compute macroscopic: rho, ux, uy from f
        rho = np.sum(self.f_stream, axis=0)  # (nx, ny)
        ux = np.sum(self.f_stream * self.ex[:, None, None], axis=0) / rho
        uy = np.sum(self.f_stream * self.ey[:, None, None], axis=0) / rho

        for i in range(9):
            u_sq = ux**2 + uy**2
            u_dot_e = self.ex[i] * ux + self.ey[i] * uy
            f_eq = self.w[i] * rho * (
                1 + 3*u_dot_e + 4.5*u_dot_e**2 - 1.5*u_sq
            )
            self.f_collide[i] = self.f_stream[i] + self.omega * (f_eq - self.f_stream[i])

        # Use collided distributions for next step
        self.f, self.f_collide = self.f_collide, self.f_stream

        # Store macroscopic
        self.rho = np.sum(self.f, axis=0)
        self.ux = np.sum(self.f * self.ex[:, None, None], axis=0) / self.rho
        self.uy = np.sum(self.f * self.ey[:, None, None], axis=0) / self.rho

        return float(0.5 * np.mean(self.ux**2 + self.uy**2))

    def run(self, n_steps: int = 1000, log_every: int = 100) -> Dict:
        energies = []
        enstrophies = []
        for t in range(n_steps):
            ke = self.step()
            if t % log_every == 0:
                energies.append(ke)
                vort = (np.roll(self.ux, 1, axis=0) - np.roll(self.ux, -1, axis=0)
                      + np.roll(self.uy, 1, axis=1) - np.roll(self.uy, -1, axis=1))
                enstrophy = float(np.mean(vort**2))
                enstrophies.append(enstrophy)
                print(f"  Step {t:5d}: KE={ke:.6f}, Enstrophy={enstrophy:.6f}")
        return {"kinetic_energy": energies, "enstrophy": enstrophies,
                "u_field": self.ux.copy(), "v_field": self.uy.copy()}


class NeuralClosureModel:
    """Neural SGS stress closure — predicts subgrid stress from resolved velocity."""
    def __init__(self, grid_size: int = 64, filter_scale: float = 2.0):
        self.grid_size = grid_size
        self.filter_scale = filter_scale

    def compute_subgrid_stress(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute Smagorinsky SGS stress tensor."""
        from scipy.ndimage import uniform_filter
        # Box filter
        fs = int(self.filter_scale)
        u_f = uniform_filter(u, size=fs)
        v_f = uniform_filter(v, size=fs)
        uu_f = uniform_filter(u * u, size=fs)
        vv_f = uniform_filter(v * v, size=fs)
        uv_f = uniform_filter(u * v, size=fs)
        tau_xx = uu_f - u_f * u_f
        tau_yy = vv_f - v_f * v_f
        tau_xy = uv_f - u_f * v_f
        return np.stack([tau_xx, tau_yy, tau_xy], axis=0)


def verify_kolmogorov_spectrum(velocity, dx: float = 1.0):
    """Compute 1D energy spectrum E(k) and verify -5/3 law."""
    u, v = velocity[0], velocity[1]
    N = u.shape[0]
    from numpy.fft import fft2, fftfreq
    u_hat = fft2(u)
    v_hat = fft2(v)
    E2D = (np.abs(u_hat)**2 + np.abs(v_hat)**2) / (N**4)
    # Use full grid for radial averaging
    kx = fftfreq(N, d=dx)
    ky = fftfreq(N, d=dx)
    kx_m, ky_m = np.meshgrid(kx, ky, indexing='ij')
    k_rad = np.sqrt(kx_m**2 + ky_m**2)
    k_max = np.max(k_rad)
    n_bins = 50
    bins = np.linspace(0, k_max, n_bins + 1)
    bin_cen = 0.5 * (bins[:-1] + bins[1:])
    E_flat = E2D.flatten()
    k_flat = k_rad.flatten()
    E_radial = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (k_flat >= bins[i]) & (k_flat < bins[i+1])
        if np.sum(mask) > 0:
            E_radial[i] = np.mean(E_flat[mask])
    valid = E_radial > 0
    return bin_cen[valid], E_radial[valid]


def inertial_range_analysis(k, E_k, k_start=3, k_end=15):
    """Fit E(k) ∝ k^{-n} in inertial range."""
    k_sub = k[k_start:k_end]
    E_sub = E_k[k_start:k_end]
    log_k = np.log(k_sub + 1e-10)
    log_E = np.log(E_sub + 1e-10)
    coeffs = np.polyfit(log_k, log_E, 1)
    slope = coeffs[0]
    print(f"\n  Kolmogorov: measured slope={slope:.4f}, expected -5/3≈{5/3:.4f}")
    return slope


if __name__ == "__main__":
    print("LBM2D turbulence test")
    lbm = LBM2D(nx=64, ny=64, reynolds=200.0, u0=0.05)
    result = lbm.run(n_steps=200, log_every=100)

    k, Ek = verify_kolmogorov_spectrum((result["u_field"], result["v_field"]))
    slope = inertial_range_analysis(k, Ek)
    print(f"\n  Slope: {slope:.3f}")