"""
Topic 1: Neural Operator Convergence Theory
FNO — robust MLX implementation.
Uses Linear layers + NumPy FFT for spectral transform.
"""
from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import numpy as np


# ─── NumPy FFT wrappers ──────────────────────────────────────
def _fft1(x, axis: int = -1):
    return np.fft.rfft(np.array(x), axis=axis)


def _ifft1(x_ft, axis: int = -1, d: int = None):
    return np.fft.irfft(np.array(x_ft), axis=axis, n=d)


def _fft2(x, axes=(-2, -1)):
    return np.fft.rfft2(np.array(x), axes=axes)


def _ifft2(x_ft, axes=(-2, -1), shape=None):
    return np.fft.irfft2(np.array(x_ft), axes=axes, s=shape)


# ─── Spectral Conv 1D ────────────────────────────────────────
class SpectralConv1d(nn.Module):
    """FFT → learnable diagonal modes → iFFT."""
    def __init__(self, sensors: int, width: int, modes: int):
        super().__init__()
        self.sensors = sensors
        self.width = width
        self.modes = modes
        self.w_re = mx.random.normal(shape=(width, modes))
        self.w_im = mx.random.normal(shape=(width, modes))

    def __call__(self, h: mx.array) -> mx.array:
        """h: (batch, sensors, width) → (batch, sensors, width)"""
        batch, sensors, width = h.shape
        h_ft = _fft1(h, axis=1)            # (batch, n_freq, width), n_freq = sensors//2+1
        n_freq = h_ft.shape[1]
        m = min(self.modes, n_freq)

        # Truncate to m modes and convert to numpy
        h_m_np = np.array(h_ft[:, :m, :])  # (batch, m, width)

        # w_r[w,m] * h_m[b,m,w] -> (batch, m, width)
        w_r = np.array(self.w_re[:width, :m])  # (width, m)
        w_i = np.array(self.w_im[:width, :m])

        real_part = h_m_np.real * w_r.T[None, :, :] - h_m_np.imag * w_i.T[None, :, :]
        imag_part = h_m_np.real * w_i.T[None, :, :] + h_m_np.imag * w_r.T[None, :, :]

        # Zero-pad to full spectrum
        h_full = np.zeros((batch, n_freq, width), dtype=np.complex128)
        h_full[:, :m, :] = real_part + 1j * imag_part

        return mx.array(_ifft1(h_full, axis=1, d=sensors))


# ─── Spectral Conv 2D ────────────────────────────────────────
class SpectralConv2d(nn.Module):
    """FFT2 → learnable diagonal modes → iFFT2."""
    def __init__(self, H: int, W: int, width: int, modes: int):
        super().__init__()
        self.H, self.W = H, W
        self.width = width
        self.modes = modes
        self.w_re = mx.random.normal(shape=(width, modes, modes))
        self.w_im = mx.random.normal(shape=(width, modes, modes))

    def __call__(self, h: mx.array) -> mx.array:
        """h: (batch, H, W, width) → (batch, H, W, width)"""
        batch = h.shape[0]
        # Move width to channel: (batch, width, H, W)
        h_chn = mx.transpose(h, (0, 3, 1, 2))

        h_ft = _fft2(h_chn, axes=(-2, -1))  # (batch, width, fH, fW)
        fH, fW = h_ft.shape[2], h_ft.shape[3]
        m = min(self.modes, fH, fW)

        h_m_np = np.array(h_ft[:, :, :m, :m])  # (batch, width, m, m)

        w_r = np.array(self.w_re[:self.width, :m, :m])  # (width, m, m)
        w_i = np.array(self.w_im[:self.width, :m, :m])

        h_new_np = (h_m_np.real * w_r[None, :, :, :] -
                    h_m_np.imag * w_i[None, :, :, :]) \
                 + 1j * (h_m_np.real * w_i[None, :, :, :] +
                         h_m_np.imag * w_r[None, :, :, :])

        # Zero-pad back to full spectrum
        h_full_np = np.zeros((batch, self.width, fH, fW), dtype=np.complex128)
        h_full_np[:, :, :m, :m] = h_new_np

        h_out = _ifft2(h_full_np, axes=(-2, -1), shape=(self.H, self.W))
        # Back to (batch, H, W, width)
        return mx.transpose(mx.array(h_out), [0, 2, 3, 1])


# ─── FNO 1D ──────────────────────────────────────────────────
class FNO1d(nn.Module):
    def __init__(self, num_sensors: int = 100, width: int = 64,
                 modes: int = 16, num_layers: int = 4):
        super().__init__()
        self.num_sensors = num_sensors
        self.width = width
        self.lifting = nn.Linear(num_sensors, width)
        self.spec = [SpectralConv1d(num_sensors, width, modes)
                     for _ in range(num_layers)]
        self.lin = [nn.Linear(width, width) for _ in range(num_layers)]
        self.projection = nn.Linear(width, 1)  # (width -> 1), applied per point

    def __call__(self, u: mx.array) -> mx.array:
        """u: (batch, num_sensors) → (batch, num_sensors)"""
        batch = u.shape[0]
        h = nn.gelu(self.lifting(u))                           # (batch, width)
        h = mx.broadcast_to(h[:, None, :],
                           (batch, self.num_sensors, self.width))

        for spec, lin in zip(self.spec, self.lin):
            h_f = spec(h)                                     # (batch, sensors, width)
            h_t = nn.gelu(lin(h_f))                           # (batch, sensors, width)
            h = h + 0.1 * h_t

        # Project: (batch, sensors, width) -> (batch, sensors, 1) -> squeeze
        out = self.projection(h)                               # (batch, sensors, 1)
        return out[:, :, 0]                                    # (batch, sensors)


# ─── FNO 2D ──────────────────────────────────────────────────
class FNO2d(nn.Module):
    def __init__(self, H: int = 64, W: int = None, width: int = 64,
                 modes: int = 12, num_layers: int = 4):
        super().__init__()
        if W is None:
            W = H
        self.H, self.W, self.width = H, W, width
        n_pts = H * W
        self.lifting = nn.Linear(n_pts, width)
        self.spec = [SpectralConv2d(H, W, width, modes)
                     for _ in range(num_layers)]
        self.lin = [nn.Linear(width, width) for _ in range(num_layers)]
        self.projection = nn.Linear(width, 1)  # (width -> 1), applied per spatial point

    def __call__(self, u: mx.array) -> mx.array:
        """u: (batch, H, W) → (batch, H, W)"""
        batch = u.shape[0]
        n_pts = self.H * self.W
        h = nn.gelu(self.lifting(u.reshape(batch, n_pts)))    # (batch, width)
        h = h.reshape(batch, 1, 1, self.width)
        h = mx.broadcast_to(h,
                           (batch, self.H, self.W, self.width))

        for spec, lin in zip(self.spec, self.lin):
            h_f = spec(h)                                      # (batch, H, W, width)
            h_t = nn.gelu(lin(h_f))                            # (batch, H, W, width)
            h = h + 0.1 * h_t

        out = self.projection(h)                              # (batch, H, W, 1)
        return out[:, :, :, 0]                                 # (batch, H, W)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/isaac/clawd/research/hermes')
    import mlx.core as mx; mx.set_default_device(mx.gpu)

    print("FNO MLX test")
    m1 = FNO1d(num_sensors=64, width=32, modes=8, num_layers=3)
    o1 = m1(mx.random.normal((4, 64)))
    print(f"1D FNO: (4,64) -> {o1.shape}")

    m2 = FNO2d(H=16, width=32, modes=6, num_layers=3)
    o2 = m2(mx.random.normal((4, 16, 16)))
    print(f"2D FNO: (4,16,16) -> {o2.shape}")