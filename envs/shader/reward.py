"""
Reward computation for the shader environment.

Provides SSIM (structural similarity) between two raw RGBA pixel buffers.
Uses scipy windowed SSIM when available, falls back to global-stats SSIM.
"""

import numpy as np

# SSIM constants (from Wang et al. 2004)
_K1 = 0.01
_K2 = 0.03
_L = 1.0  # dynamic range for float images in [0, 1]
_C1 = (_K1 * _L) ** 2
_C2 = (_K2 * _L) ** 2

try:
    from scipy.ndimage import uniform_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _channel(a: np.ndarray, b: np.ndarray, window: int = 11) -> float:
    """SSIM for a single channel (H, W) float array in [0, 1]."""
    if _HAS_SCIPY:
        mu_a = uniform_filter(a, size=window)
        mu_b = uniform_filter(b, size=window)
        sigma_aa = uniform_filter(a * a, size=window) - mu_a * mu_a
        sigma_bb = uniform_filter(b * b, size=window) - mu_b * mu_b
        sigma_ab = uniform_filter(a * b, size=window) - mu_a * mu_b
    else:
        mu_a = np.mean(a)
        mu_b = np.mean(b)
        sigma_aa = np.var(a)
        sigma_bb = np.var(b)
        sigma_ab = np.mean((a - mu_a) * (b - mu_b))

    num = (2 * mu_a * mu_b + _C1) * (2 * sigma_ab + _C2)
    den = (mu_a ** 2 + mu_b ** 2 + _C1) * (sigma_aa + sigma_bb + _C2)

    ssim_map = num / den

    if isinstance(ssim_map, np.ndarray):
        return float(np.mean(ssim_map))
    return float(ssim_map)


def ssim(ref: bytes | None, agent: bytes | None, width: int, height: int) -> float:
    """
    Mean SSIM between two raw RGBA byte buffers (top-left origin).

    Drops alpha channel, computes per-channel SSIM on RGB, returns mean.
    Returns 0.0 on malformed input.
    """
    if ref is None or agent is None:
        return 0.0

    expected = width * height * 4
    if len(ref) != expected or len(agent) != expected:
        return 0.0

    ref_arr = np.frombuffer(ref, dtype=np.uint8).reshape(height, width, 4)
    agent_arr = np.frombuffer(agent, dtype=np.uint8).reshape(height, width, 4)

    # Drop alpha, convert to float [0, 1]
    ref_rgb = ref_arr[:, :, :3].astype(np.float64) / 255.0
    agent_rgb = agent_arr[:, :, :3].astype(np.float64) / 255.0

    # Per-channel SSIM, averaged and clamped to [0, 1]
    scores = []
    for c in range(3):
        scores.append(_channel(ref_rgb[:, :, c], agent_rgb[:, :, c]))

    return float(np.clip(np.mean(scores), 0.0, 1.0))
