"""
Audio-reactive strange attractor renderer.

Simulates thousands of particles orbiting two deterministic strange attractors
(Lorenz + Rössler) in 3D. Particles leave neon-colored trails in a persistent
accumulation buffer that fades over time, creating cinematic glowing trail-ribbons.

Audio mapping (Phase 1 audio intelligence):
  IMMEDIATE / PERCUSSIVE (fast lerp ~0.6–0.8):
  - sub_bass           → Lorenz σ wide-range modulation + scale pulse (cloud breathes)
  - percussive_impact  → Lorenz ρ strong lobing + brightness pulse + beat shockwave
  - brilliance         → hi-hat sparkle reseed + hue shimmer
  - spectral_flux      → camera elevation agitation
  - timbre_velocity    → extra Lorenz σ turbulence during timbral transitions (C1)

  TONAL / HARMONIC (medium-slow lerp ~0.2):
  - harmonic_energy    → Rössler a spiral tightness
  - global_energy      → Rössler c bifurcation + simulation speed + adaptive trail decay
  - key_stability      → additional Rössler c chaos (atonal=wilder) + trail length (C4)

  PITCH / COLOR (slow lerp ~0.04–0.12):
  - pitch_register     → camera elevation arc: bass=level, treble=looking up (C3)
  - pitch_hue          → dominant chord hue; lerp rate modulated by key_stability (C4)
  - spectral_centroid  → fine hue sparkle drift

  BEAT / RHYTHM (C5):
  - is_beat            → shockwave: reseed + camera kick + flash bloom
  - is_downbeat        → 1.6× flash, 1.8× camera kick, higher reseed (bar 1 hits hardest)
  - beat_position      → bar-breathing: cloud contracts/expands with the bar grid
  - spectral_flatness  → morph blend weight (noisy audio = more Rössler)

  STRUCTURE (C2):
  - section_change     → mass reseed (40%) + hue identity jump + bloom flash
  - section_index      → each section gets a golden-ratio-spaced palette hue

  ONSET SHAPE (W4):
  - onset_type         → "harmonic" skips reseed; "transient" adds azimuth micro-jolt
  - onset_sharpness    → scales transient azimuth jolt magnitude
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from chromascope.experiment.base import BaseConfig, BaseVisualizer
from chromascope.experiment.colorgrade import chromatic_aberration, vignette

# ---------------------------------------------------------------------------
# Optional Numba acceleration
# ---------------------------------------------------------------------------
try:
    import numba as _numba

    _NUMBA_OK: bool = True
except ImportError:  # pragma: no cover
    _numba = None  # type: ignore[assignment]
    _NUMBA_OK = False

# ---------------------------------------------------------------------------
# Palette anchors: (h0, h1, s, v) per attractor type
# Hue linearly interpolated with z_depth: hue = h0 + (h1-h0) * z_depth
# ---------------------------------------------------------------------------
_ATTRACTOR_PALETTES: Dict[str, Dict[str, Tuple[float, float, float, float]]] = {
    "neon_aurora": {
        "lorenz":  (0.50, 0.83, 1.0, 1.0),   # cyan → magenta
        "rossler": (0.25, 0.55, 1.0, 0.95),  # electric lime → ice blue
    },
    "plasma_coil": {
        "lorenz":  (0.06, 0.92, 1.0, 1.0),   # hot orange → hot pink
        "rossler": (0.17, 0.38, 1.0, 1.0),   # neon yellow → neon green
    },
    "void_fire": {
        "lorenz":  (0.63, 0.53, 1.0, 1.0),   # deep blue → cyan
        "rossler": (0.75, 0.14, 1.0, 1.0),   # deep purple → gold
    },
    "quantum_foam": {
        "lorenz":  (0.33, 0.53, 1.0, 1.0),   # electric green → cyan
        "rossler": (0.92, 0.03, 1.0, 1.0),   # hot magenta → neon red
    },
}

# ---------------------------------------------------------------------------
# Numba-accelerated kernels (compiled on first call)
# ---------------------------------------------------------------------------
if _NUMBA_OK:

    @_numba.njit(parallel=True, fastmath=True, cache=True)  # type: ignore[misc]
    def _rk4_lorenz_kernel(
        pts: np.ndarray,
        sigma: float,
        rho: float,
        beta: float,
        dt: float,
        substeps: int,
    ) -> None:
        """In-place RK4 integration of N Lorenz particles. Thread-safe: each
        particle is independent (prange over rows)."""
        N = pts.shape[0]
        for i in _numba.prange(N):  # type: ignore[attr-defined]
            x = pts[i, 0]
            y = pts[i, 1]
            z = pts[i, 2]
            for _ in range(substeps):
                k1x = sigma * (y - x)
                k1y = x * (rho - z) - y
                k1z = x * y - beta * z

                x2 = x + 0.5 * dt * k1x
                y2 = y + 0.5 * dt * k1y
                z2 = z + 0.5 * dt * k1z
                k2x = sigma * (y2 - x2)
                k2y = x2 * (rho - z2) - y2
                k2z = x2 * y2 - beta * z2

                x3 = x + 0.5 * dt * k2x
                y3 = y + 0.5 * dt * k2y
                z3 = z + 0.5 * dt * k2z
                k3x = sigma * (y3 - x3)
                k3y = x3 * (rho - z3) - y3
                k3z = x3 * y3 - beta * z3

                x4 = x + dt * k3x
                y4 = y + dt * k3y
                z4 = z + dt * k3z
                k4x = sigma * (y4 - x4)
                k4y = x4 * (rho - z4) - y4
                k4z = x4 * y4 - beta * z4

                x = x + dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
                y = y + dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
                z = z + dt * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0
            pts[i, 0] = x
            pts[i, 1] = y
            pts[i, 2] = z

    @_numba.njit(parallel=True, fastmath=True, cache=True)  # type: ignore[misc]
    def _rk4_rossler_kernel(
        pts: np.ndarray,
        a: float,
        b: float,
        c: float,
        dt: float,
        substeps: int,
    ) -> None:
        """In-place RK4 integration of N Rössler particles."""
        N = pts.shape[0]
        for i in _numba.prange(N):  # type: ignore[attr-defined]
            x = pts[i, 0]
            y = pts[i, 1]
            z = pts[i, 2]
            for _ in range(substeps):
                k1x = -y - z
                k1y = x + a * y
                k1z = b + z * (x - c)

                x2 = x + 0.5 * dt * k1x
                y2 = y + 0.5 * dt * k1y
                z2 = z + 0.5 * dt * k1z
                k2x = -y2 - z2
                k2y = x2 + a * y2
                k2z = b + z2 * (x2 - c)

                x3 = x + 0.5 * dt * k2x
                y3 = y + 0.5 * dt * k2y
                z3 = z + 0.5 * dt * k2z
                k3x = -y3 - z3
                k3y = x3 + a * y3
                k3z = b + z3 * (x3 - c)

                x4 = x + dt * k3x
                y4 = y + dt * k3y
                z4 = z + dt * k3z
                k4x = -y4 - z4
                k4y = x4 + a * y4
                k4z = b + z4 * (x4 - c)

                x = x + dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
                y = y + dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
                z = z + dt * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0
            pts[i, 0] = x
            pts[i, 1] = y
            pts[i, 2] = z

    @_numba.njit(cache=True)  # type: ignore[misc]
    def _splat_glow_kernel(
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        r_arr: np.ndarray,
        g_arr: np.ndarray,
        b_arr: np.ndarray,
        weight_arr: np.ndarray,
        accum: np.ndarray,
        H: int,
        W: int,
    ) -> None:
        """Bilinear scatter of N particles onto RGB accum buffer (serial — safe)."""
        N = x_arr.shape[0]
        for i in range(N):
            px = x_arr[i]
            py = y_arr[i]
            x0 = int(math.floor(px))
            y0 = int(math.floor(py))
            fx = float(px - x0)
            fy = float(py - y0)
            ri = r_arr[i] * weight_arr[i]
            gi = g_arr[i] * weight_arr[i]
            bi = b_arr[i] * weight_arr[i]
            for dy in range(2):
                wy = fy if dy else (1.0 - fy)
                yi = y0 + dy
                if yi < 0 or yi >= H:
                    continue
                for dx in range(2):
                    wx = fx if dx else (1.0 - fx)
                    xi = x0 + dx
                    if xi < 0 or xi >= W:
                        continue
                    w = wx * wy
                    accum[yi, xi, 0] += ri * w
                    accum[yi, xi, 1] += gi * w
                    accum[yi, xi, 2] += bi * w

    @_numba.njit(cache=True)  # type: ignore[misc]
    def _splat_soft_kernel(
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        r_arr: np.ndarray,
        g_arr: np.ndarray,
        b_arr: np.ndarray,
        weight_arr: np.ndarray,
        accum: np.ndarray,
        H: int,
        W: int,
        splat_sigma: float,
    ) -> None:
        """Gaussian-disc scatter: each particle deposits a soft circular footprint.

        Serial over particles (safe for accumulation without atomics).
        Kernel radius = ceil(2.5 * sigma) pixels; weights fall off as exp(-r²/2σ²).
        """
        N = x_arr.shape[0]
        r = int(math.ceil(splat_sigma * 2.5))
        inv2sig2 = 0.5 / (splat_sigma * splat_sigma)
        for i in range(N):
            xi = int(math.floor(x_arr[i] + 0.5))
            yi = int(math.floor(y_arr[i] + 0.5))
            ri = r_arr[i] * weight_arr[i]
            gi = g_arr[i] * weight_arr[i]
            bi = b_arr[i] * weight_arr[i]
            for dy in range(-r, r + 1):
                yw = yi + dy
                if yw < 0 or yw >= H:
                    continue
                dy2 = float(dy * dy)
                for dx in range(-r, r + 1):
                    xw = xi + dx
                    if xw < 0 or xw >= W:
                        continue
                    g = math.exp(-(dy2 + float(dx * dx)) * inv2sig2)
                    accum[yw, xw, 0] += ri * g
                    accum[yw, xw, 1] += gi * g
                    accum[yw, xw, 2] += bi * g


# ---------------------------------------------------------------------------
# NumPy fallbacks (always defined, used when Numba is unavailable)
# ---------------------------------------------------------------------------

def _rk4_lorenz_numpy(
    pts: np.ndarray,
    sigma: float,
    rho: float,
    beta: float,
    dt: float,
    substeps: int,
) -> None:
    """Vectorized NumPy RK4 for Lorenz system (in-place)."""
    for _ in range(substeps):
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        k1x = sigma * (y - x)
        k1y = x * (rho - z) - y
        k1z = x * y - beta * z

        x2 = x + 0.5 * dt * k1x
        y2 = y + 0.5 * dt * k1y
        z2 = z + 0.5 * dt * k1z
        k2x = sigma * (y2 - x2)
        k2y = x2 * (rho - z2) - y2
        k2z = x2 * y2 - beta * z2

        x3 = x + 0.5 * dt * k2x
        y3 = y + 0.5 * dt * k2y
        z3 = z + 0.5 * dt * k2z
        k3x = sigma * (y3 - x3)
        k3y = x3 * (rho - z3) - y3
        k3z = x3 * y3 - beta * z3

        x4 = x + dt * k3x
        y4 = y + dt * k3y
        z4 = z + dt * k3z
        k4x = sigma * (y4 - x4)
        k4y = x4 * (rho - z4) - y4
        k4z = x4 * y4 - beta * z4

        pts[:, 0] += dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
        pts[:, 1] += dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
        pts[:, 2] += dt * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0


def _rk4_rossler_numpy(
    pts: np.ndarray,
    a: float,
    b: float,
    c: float,
    dt: float,
    substeps: int,
) -> None:
    """Vectorized NumPy RK4 for Rössler system (in-place)."""
    for _ in range(substeps):
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        k1x = -y - z
        k1y = x + a * y
        k1z = b + z * (x - c)

        x2 = x + 0.5 * dt * k1x
        y2 = y + 0.5 * dt * k1y
        z2 = z + 0.5 * dt * k1z
        k2x = -y2 - z2
        k2y = x2 + a * y2
        k2z = b + z2 * (x2 - c)

        x3 = x + 0.5 * dt * k2x
        y3 = y + 0.5 * dt * k2y
        z3 = z + 0.5 * dt * k2z
        k3x = -y3 - z3
        k3y = x3 + a * y3
        k3z = b + z3 * (x3 - c)

        x4 = x + dt * k3x
        y4 = y + dt * k3y
        z4 = z + dt * k3z
        k4x = -y4 - z4
        k4y = x4 + a * y4
        k4z = b + z4 * (x4 - c)

        pts[:, 0] += dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
        pts[:, 1] += dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
        pts[:, 2] += dt * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0


def _splat_glow_numpy(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    r_arr: np.ndarray,
    g_arr: np.ndarray,
    b_arr: np.ndarray,
    weight_arr: np.ndarray,
    accum: np.ndarray,
    H: int,
    W: int,
) -> None:
    """Bilinear scatter via np.add.at (NumPy fallback)."""
    x0 = np.floor(x_arr).astype(np.int64)
    y0 = np.floor(y_arr).astype(np.int64)
    fx = (x_arr - x0).astype(np.float32)
    fy = (y_arr - y0).astype(np.float32)
    rgb = np.stack(
        [r_arr * weight_arr, g_arr * weight_arr, b_arr * weight_arr], axis=1
    ).astype(np.float32)

    for dy in range(2):
        wy = fy if dy else (1.0 - fy)
        for dx in range(2):
            wx = fx if dx else (1.0 - fx)
            w = (wx * wy).astype(np.float32)
            xi = x0 + dx
            yi = y0 + dy
            valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            if not np.any(valid):
                continue
            np.add.at(
                accum,
                (yi[valid], xi[valid]),
                rgb[valid] * w[valid, np.newaxis],
            )


def _splat_soft_numpy(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    r_arr: np.ndarray,
    g_arr: np.ndarray,
    b_arr: np.ndarray,
    weight_arr: np.ndarray,
    accum: np.ndarray,
    H: int,
    W: int,
    splat_sigma: float,
) -> None:
    """Gaussian-disc scatter (NumPy fallback for _splat_soft_kernel)."""
    r = int(math.ceil(splat_sigma * 2.5))
    inv2sig2 = 0.5 / (splat_sigma * splat_sigma)
    xi = np.round(x_arr).astype(np.int64)
    yi = np.round(y_arr).astype(np.int64)
    rgb = np.stack(
        [r_arr * weight_arr, g_arr * weight_arr, b_arr * weight_arr], axis=1
    ).astype(np.float32)

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            g = math.exp(-(dx * dx + dy * dy) * inv2sig2)
            if g < 1e-6:
                continue
            xw = xi + dx
            yw = yi + dy
            valid = (xw >= 0) & (xw < W) & (yw >= 0) & (yw < H)
            if not np.any(valid):
                continue
            np.add.at(
                accum,
                (yw[valid], xw[valid]),
                rgb[valid] * float(g),
            )


# ---------------------------------------------------------------------------
# Module-level dispatch: call Numba kernels if available, else NumPy
# ---------------------------------------------------------------------------
if _NUMBA_OK:

    def rk4_lorenz(
        pts: np.ndarray,
        sigma: float,
        rho: float,
        beta: float,
        dt: float,
        substeps: int,
    ) -> None:
        _rk4_lorenz_kernel(pts, sigma, rho, beta, dt, substeps)

    def rk4_rossler(
        pts: np.ndarray,
        a: float,
        b: float,
        c: float,
        dt: float,
        substeps: int,
    ) -> None:
        _rk4_rossler_kernel(pts, a, b, c, dt, substeps)

    def splat_glow(
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        r_arr: np.ndarray,
        g_arr: np.ndarray,
        b_arr: np.ndarray,
        weight_arr: np.ndarray,
        accum: np.ndarray,
        H: int,
        W: int,
    ) -> None:
        _splat_glow_kernel(x_arr, y_arr, r_arr, g_arr, b_arr, weight_arr, accum, H, W)

    def splat_soft(
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        r_arr: np.ndarray,
        g_arr: np.ndarray,
        b_arr: np.ndarray,
        weight_arr: np.ndarray,
        accum: np.ndarray,
        H: int,
        W: int,
        splat_sigma: float,
    ) -> None:
        _splat_soft_kernel(
            x_arr, y_arr, r_arr, g_arr, b_arr, weight_arr, accum, H, W, splat_sigma
        )

else:  # pragma: no cover

    def rk4_lorenz(pts, sigma, rho, beta, dt, substeps):  # type: ignore[misc]
        _rk4_lorenz_numpy(pts, sigma, rho, beta, dt, substeps)

    def rk4_rossler(pts, a, b, c, dt, substeps):  # type: ignore[misc]
        _rk4_rossler_numpy(pts, a, b, c, dt, substeps)

    def splat_glow(x_arr, y_arr, r_arr, g_arr, b_arr, weight_arr, accum, H, W):  # type: ignore[misc]
        _splat_glow_numpy(x_arr, y_arr, r_arr, g_arr, b_arr, weight_arr, accum, H, W)

    def splat_soft(x_arr, y_arr, r_arr, g_arr, b_arr, weight_arr, accum, H, W, splat_sigma):  # type: ignore[misc]
        _splat_soft_numpy(
            x_arr, y_arr, r_arr, g_arr, b_arr, weight_arr, accum, H, W, splat_sigma
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AttractorConfig(BaseConfig):
    """Configuration for the strange attractor renderer."""

    # Particle simulation
    num_particles: int = 3000
    lorenz_sigma: float = 10.0
    lorenz_rho: float = 28.0
    lorenz_beta: float = 2.6667
    rossler_a: float = 0.2
    rossler_b: float = 0.2
    rossler_c: float = 5.7

    # Rendering behaviour
    blend_mode: str = "dual"       # lorenz | rossler | dual | morph
    trail_decay: float = 0.96      # [0, 1] — how fast trails fade
    substeps: int = 6              # RK4 sub-steps per frame
    projection_speed: float = 0.2  # base azimuth rotation speed (rad/s)

    # Visuals
    attractor_palette: str = "neon_aurora"
    glow_radius: float = 2.0       # sigma for gaussian bloom pass (overrides BaseConfig int)
    particle_brightness: float = 1.5  # HDR exposure multiplier
    splat_radius: float = 2.0      # per-particle gaussian disc sigma (px); 0 = bilinear fallback

    # Audio responsiveness
    audio_sensitivity: float = 1.0    # global multiplier for all audio effects
    beat_flash_strength: float = 3.0  # peak brightness multiplier on beat (clipped to [0.5, 3])
    pitch_color_strength: float = 0.5 # fraction of pitch_hue applied to palette hue offset


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def _rotation_matrix(az: float, el: float) -> np.ndarray:
    """3×3 rotation matrix: azimuth (around Y) then elevation (around X)."""
    ca, sa = math.cos(az), math.sin(az)
    ce, se = math.cos(el), math.sin(el)
    Ry = np.array(
        [[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]], dtype=np.float64
    )
    Rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, ce, -se], [0.0, se, ce]], dtype=np.float64
    )
    return Rx @ Ry


def _hsv_to_rgb_batch(
    h: np.ndarray, s: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Vectorized HSV → RGB for 1D arrays of N particles. Returns (N,3) float32."""
    h6 = (h * 6.0) % 6.0
    idx = h6.astype(np.int32)
    f = (h6 - idx).astype(np.float32)
    p = (v * (1.0 - s)).astype(np.float32)
    q = (v * (1.0 - s * f)).astype(np.float32)
    t = (v * (1.0 - s * (1.0 - f))).astype(np.float32)
    v = v.astype(np.float32)

    N = len(h)
    rgb = np.zeros((N, 3), dtype=np.float32)
    for mask_val, r_src, g_src, b_src in [
        (0, v, t, p),
        (1, q, v, p),
        (2, p, v, t),
        (3, p, q, v),
        (4, t, p, v),
        (5, v, p, q),
    ]:
        m = idx == mask_val
        if np.any(m):
            rgb[m, 0] = r_src[m]
            rgb[m, 1] = g_src[m]
            rgb[m, 2] = b_src[m]
    return rgb


class AttractorRenderer(BaseVisualizer):
    """
    Renders audio-reactive Lorenz and Rössler strange attractors.

    Particles orbit the attractors in 3D; their 3D positions are projected onto
    a 2D screen and splatted as colored blobs onto a persistent float32
    accumulation buffer. The buffer fades each frame (trail_decay) producing
    glowing trail-ribbons. HDR Reinhard tone-mapping and a gaussian bloom pass
    create the neon cinematic look.
    """

    _DT_SIM: float = 0.01  # simulation timestep per frame

    def __init__(
        self,
        config: Optional[AttractorConfig] = None,
        seed: Optional[int] = None,
        center_pos: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__(config or AttractorConfig(), seed, center_pos)
        self.cfg: AttractorConfig = self.cfg  # type annotation

        H, W = self.cfg.height, self.cfg.width
        N = self.cfg.num_particles

        # Particle positions — float64 for numerical stability
        self._lorenz_pts: np.ndarray = np.empty((N, 3), dtype=np.float64)
        self._rossler_pts: np.ndarray = np.empty((N, 3), dtype=np.float64)

        # Persistent HDR accumulation buffer (H, W, 3) float32
        self._accum: np.ndarray = np.zeros((H, W, 3), dtype=np.float32)

        # Projection state
        self._proj_az: float = 0.0
        self._proj_el: float = 0.3

        # Hue drift
        self._hue_offset: float = 0.0

        # Normalization constants (filled by _warmup_and_normalize)
        self._lorenz_norm: Tuple[np.ndarray, float] = (np.zeros(3), 1.0)
        self._rossler_norm: Tuple[np.ndarray, float] = (np.zeros(3), 1.0)

        # Audio-reactive state (beyond base smoothed values)
        self._beat_flash: float = 0.0      # decaying beat flash; set on beat, fades ~5 frames
        self._scale_pulse: float = 1.0     # sub-bass driven projection scale [0.8, 1.4]
        self._brightness_pulse: float = 1.0  # percussive particle brightness [1.0, 3.0]
        self._hue_target: float = 0.0      # pitch-derived hue target (lerped slowly)

        # Phase 1 audio intelligence state
        self._smooth_pitch_register: float = 0.5   # C3: melodic register [0=bass, 1=treble]
        self._smooth_key_stability: float = 0.8    # C4: harmonic stability [0=atonal, 1=stable]
        self._smooth_timbre_velocity: float = 0.0  # C1: rate of timbral change
        self._smooth_bandwidth: float = 0.5        # C7: normalized spectral bandwidth
        self._section_index: int = 0               # C2: last seen section number

        # Initialise particles and compute normalization
        self._warmup_and_normalize()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _warmup_and_normalize(self) -> None:
        """Seed particles, run 2000 integration steps, compute normalization."""
        N = self.cfg.num_particles
        dt = self._DT_SIM

        # Lorenz: start near centre with small perturbations
        self._lorenz_pts[:] = self.rng.normal(0.0, 0.5, (N, 3))
        self._lorenz_pts[:, 2] += 25.0  # shift toward typical Z centre

        # Rössler: start near attractor entrance
        self._rossler_pts[:] = self.rng.normal(0.0, 0.5, (N, 3))
        self._rossler_pts[:, 2] += 3.0

        # Warm up — many steps to settle onto the attractor
        n_warmup = 2000
        rk4_lorenz(
            self._lorenz_pts,
            self.cfg.lorenz_sigma,
            self.cfg.lorenz_rho,
            self.cfg.lorenz_beta,
            dt,
            n_warmup,
        )
        rk4_rossler(
            self._rossler_pts,
            self.cfg.rossler_a,
            self.cfg.rossler_b,
            self.cfg.rossler_c,
            dt,
            n_warmup,
        )

        # Safety: reset any escaped particles
        self._fix_nans(self._lorenz_pts, np.array([0.0, 0.0, 25.0]))
        self._fix_nans(self._rossler_pts, np.array([0.0, 0.0, 3.0]))

        # Record normalization from current cloud bounds
        self._lorenz_norm = self._compute_norm(self._lorenz_pts)
        self._rossler_norm = self._compute_norm(self._rossler_pts)

    def _compute_norm(
        self, pts: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Return (center, scale) such that (pts - center)/scale ≈ [-1, 1]."""
        center = pts.mean(axis=0)
        scale = float(np.abs(pts - center).max()) * 1.1
        return center, max(scale, 1e-6)

    def _fix_nans(self, pts: np.ndarray, fallback: np.ndarray) -> None:
        """Reset any NaN/Inf rows to the fallback position."""
        bad = ~np.isfinite(pts).all(axis=1)
        n_bad = int(bad.sum())
        if n_bad:
            pts[bad] = fallback + self.rng.normal(0.0, 0.1, (n_bad, 3))

    def _reseed_fraction(
        self,
        pts: np.ndarray,
        frac: float,
        norm_center: np.ndarray,
        norm_scale: float,
    ) -> None:
        """Re-scatter `frac` fraction of particles near the attractor centre."""
        N = len(pts)
        n_reseed = max(1, int(N * frac))
        indices = self.rng.integers(0, N, n_reseed)
        pts[indices] = norm_center + self.rng.normal(
            0.0, norm_scale * 0.1, (n_reseed, 3)
        )

    # ------------------------------------------------------------------
    # Audio smoothing override (faster than base class)
    # ------------------------------------------------------------------

    def _smooth_audio(self, frame_data: Dict[str, Any]) -> None:
        """Override base with faster lerp — attractors demand immediacy.

        The polisher already applies 200-300ms release envelopes. Adding the
        base class's slow=0.06 lerp on top creates ~500ms total lag — enough
        to miss an entire beat. We bypass that by using much faster rates here.
        """
        is_beat = frame_data.get("is_beat", False)
        # fast: sub-bass, percussive, brilliance, flux — these are transients
        fast = 0.80 if is_beat else 0.55
        # med: flatness, sharpness
        med = 0.30
        # slow: energy, harmonic, centroid — sustained signals
        slow = 0.20

        self._smooth_energy = self._lerp(
            self._smooth_energy, frame_data.get("global_energy", 0.1), slow
        )
        self._smooth_percussive = self._lerp(
            self._smooth_percussive, frame_data.get("percussive_impact", 0.0), fast
        )
        self._smooth_harmonic = self._lerp(
            self._smooth_harmonic, frame_data.get("harmonic_energy", 0.2), slow
        )
        self._smooth_low = self._lerp(
            self._smooth_low, frame_data.get("low_energy", 0.1), slow
        )
        self._smooth_high = self._lerp(
            self._smooth_high, frame_data.get("high_energy", 0.1), slow
        )
        self._smooth_flux = self._lerp(
            self._smooth_flux, frame_data.get("spectral_flux", 0.0), fast
        )
        self._smooth_flatness = self._lerp(
            self._smooth_flatness, frame_data.get("spectral_flatness", 0.0), med
        )
        self._smooth_sharpness = self._lerp(
            self._smooth_sharpness, frame_data.get("sharpness", 0.0), med
        )
        self._smooth_sub_bass = self._lerp(
            self._smooth_sub_bass, frame_data.get("sub_bass", 0.0), fast
        )
        self._smooth_brilliance = self._lerp(
            self._smooth_brilliance, frame_data.get("brilliance", 0.0), fast
        )
        self._smooth_centroid = self._lerp(
            self._smooth_centroid, frame_data.get("spectral_centroid", 0.5), slow
        )

        # ── Phase 1 fields ────────────────────────────────────────────────
        # pitch_register (C3): actual F0 register — slow, melodic signal
        self._smooth_pitch_register = self._lerp(
            self._smooth_pitch_register,
            float(frame_data.get("pitch_register") or 0.5), 0.06,
        )
        # key_stability (C4): how strongly a key center is held
        self._smooth_key_stability = self._lerp(
            self._smooth_key_stability,
            float(frame_data.get("key_stability") or 0.8), 0.05,
        )
        # timbre_velocity (C1): speed of timbral change — treat like a fast signal
        self._smooth_timbre_velocity = self._lerp(
            self._smooth_timbre_velocity,
            float(frame_data.get("timbre_velocity") or 0.0), 0.30,
        )
        # bandwidth_norm (C7): spectral width of the signal
        self._smooth_bandwidth = self._lerp(
            self._smooth_bandwidth,
            float(frame_data.get("bandwidth_norm") or 0.5), 0.15,
        )

    # ------------------------------------------------------------------
    # Audio-driven parameter modulation
    # ------------------------------------------------------------------

    def _modulate_params(
        self,
    ) -> Tuple[float, float, float, float, float, float]:
        """Return audio-modulated (sigma, rho, beta, a, b, c)."""
        cfg = self.cfg
        s = cfg.audio_sensitivity

        # Lorenz σ: sub_bass drives wide-range expansion/contraction
        # timbre_velocity (C1) adds turbulence during timbral transitions
        sigma = cfg.lorenz_sigma * (0.4 + self._smooth_sub_bass * 1.2 * s
                                     + self._smooth_timbre_velocity * 0.3 * s)

        # Lorenz ρ: percussive impact drives strong lobing distortion
        #   range: [14.0, 42.0] at s=1.0 (old was barely ±30%)
        rho = cfg.lorenz_rho * (0.5 + self._smooth_percussive * 1.0 * s)

        beta = cfg.lorenz_beta  # keep stable — beta changes shape fundamentally

        # Rössler a: harmonic content drives spiral tightness
        a = cfg.rossler_a * (0.5 + self._smooth_harmonic * 2.5 * s)

        b = cfg.rossler_b  # keep stable

        # Rössler c: THIS IS THE BIFURCATION PARAMETER.
        # c < 4: simple limit cycles; c ≈ 5.7: classic chaos; c > 6: wilder
        # energy → chaos; key_instability (C4) adds extra bifurcation pressure:
        # chromatic/atonal passages push the attractor into wilder regimes.
        c = cfg.rossler_c * (0.6 + self._smooth_energy * 0.8 * s
                              + (1.0 - self._smooth_key_stability) * 0.3 * s)

        # Clamp to numerically safe chaotic regimes
        sigma = float(np.clip(sigma, 2.0, 25.0))
        rho = float(np.clip(rho, 10.0, 45.0))
        beta = float(np.clip(beta, 0.5, 5.0))
        a = float(np.clip(a, 0.05, 0.45))
        b = float(np.clip(b, 0.05, 0.4))
        c = float(np.clip(c, 2.0, 10.0))

        return sigma, rho, beta, a, b, c

    # ------------------------------------------------------------------
    # 3-D projection and colour
    # ------------------------------------------------------------------

    def _project_3d_2d(
        self,
        pts: np.ndarray,
        norm: Tuple[np.ndarray, float],
        az: float,
        el: float,
        W: int,
        H: int,
        scale_mult: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalise → rotate → map to pixel coords.

        Args:
            scale_mult: Additional scale factor (bass pulse breathing).

        Returns:
            x_px, y_px : float64 pixel coordinates
            z_depth    : float32 depth in [0, 1]
        """
        center, scale = norm
        pts_n = (pts - center) / scale  # (N, 3), roughly in [-1, 1]

        rot = _rotation_matrix(az, el)
        pts_rot = pts_n @ rot.T  # (N, 3)

        # Scale to maintain aspect ratio; scale_mult breathes with bass
        half_side = min(W, H) / 2.0 * 0.85 * scale_mult
        cx = W / 2.0
        cy = H / 2.0

        x_px = cx + pts_rot[:, 0] * half_side
        y_px = cy - pts_rot[:, 1] * half_side  # flip Y for screen coords
        z_depth = np.clip((pts_rot[:, 2] + 1.0) / 2.0, 0.0, 1.0).astype(
            np.float32
        )
        return x_px, y_px, z_depth

    def _compute_colors(
        self,
        z_depth: np.ndarray,
        attractor_type: str,
        palette_name: str,
    ) -> np.ndarray:
        """Map z_depth → neon RGB via palette gradient + hue offset.

        Returns (N, 3) float32 RGB in [0, 1].
        """
        pal = _ATTRACTOR_PALETTES.get(
            palette_name, _ATTRACTOR_PALETTES["neon_aurora"]
        )[attractor_type]
        h0, h1, s_val, v_val = pal
        N = len(z_depth)
        hue = ((h0 + (h1 - h0) * z_depth + self._hue_offset) % 1.0).astype(
            np.float32
        )
        s = np.full(N, s_val, dtype=np.float32)
        v = np.full(N, v_val, dtype=np.float32)
        return _hsv_to_rgb_batch(hue, s, v)

    # ------------------------------------------------------------------
    # Splat helper
    # ------------------------------------------------------------------

    def _project_and_splat(
        self,
        pts: np.ndarray,
        norm: Tuple[np.ndarray, float],
        attractor_type: str,
        palette_name: str,
        H: int,
        W: int,
        scale_mult: float = 1.0,
        brightness_mult: float = 1.0,
    ) -> None:
        """Project particles to 2D, compute colours, splat onto _accum.

        Args:
            scale_mult:      Bass-driven projection scale (cloud breathing).
            brightness_mult: Percussive particle brightness multiplier.
        """
        x_px, y_px, z_depth = self._project_3d_2d(
            pts, norm, self._proj_az, self._proj_el, W, H, scale_mult
        )
        rgb = self._compute_colors(z_depth, attractor_type, palette_name)
        # Depth-based brightness * percussive multiplier
        weights = (
            self.cfg.particle_brightness * brightness_mult * (0.3 + 0.7 * z_depth)
        ).astype(np.float32)

        splat_soft(
            x_px.astype(np.float64),
            y_px.astype(np.float64),
            rgb[:, 0],
            rgb[:, 1],
            rgb[:, 2],
            weights,
            self._accum,
            H,
            W,
            float(self.cfg.splat_radius),
        )

    # ------------------------------------------------------------------
    # BaseVisualizer interface
    # ------------------------------------------------------------------

    def update(self, frame_data: Dict[str, Any]) -> None:
        """Advance simulation by one frame: integrate, project, splat."""
        dt = 1.0 / self.cfg.fps
        self.time += dt
        self._smooth_audio(frame_data)

        is_beat = frame_data.get("is_beat", False)
        H, W = self.cfg.height, self.cfg.width
        palette = self.cfg.attractor_palette
        mode = self.cfg.blend_mode
        s = self.cfg.audio_sensitivity

        # Phase 1 event fields
        is_downbeat    = bool(frame_data.get("is_downbeat", False))
        beat_position  = float(frame_data.get("beat_position") or 0.0)
        section_change = bool(frame_data.get("section_change", False))
        section_index  = int(frame_data.get("section_index") or 0)
        onset_type     = str(frame_data.get("onset_type") or "percussive")
        onset_sharpness = float(frame_data.get("onset_sharpness") or 0.5)

        # ── 1. BEAT FLASH ────────────────────────────────────────────────
        # Downbeats (C5: bar 1) get 1.6× the flash of regular beats — the
        # musical hierarchy is now reflected in the visual intensity hierarchy.
        if is_beat:
            flash_strength = (
                self.cfg.beat_flash_strength * 1.6 if is_downbeat
                else self.cfg.beat_flash_strength
            )
            flash_val = float(np.clip(
                self._smooth_percussive * flash_strength, 0.5, 4.0,
            ))
            self._beat_flash = max(self._beat_flash, flash_val)
        self._beat_flash *= 0.80  # ~5-frame half-life at 30fps, ~10 at 60fps

        # ── 1b. SECTION CHANGE (C2) ──────────────────────────────────────
        # New musical section = distinct hue identity + mass particle reseed.
        # Visually marks musical structure the way a film cut marks a scene change.
        if section_change and section_index != self._section_index:
            self._section_index = section_index
            # Golden-ratio-spaced hue so adjacent sections always contrast
            self._hue_target = (section_index * 0.382) % 1.0
            self._beat_flash = max(self._beat_flash, 2.0)
            frac = 0.40
            if mode in ("lorenz", "dual", "morph"):
                self._reseed_fraction(self._lorenz_pts, frac, *self._lorenz_norm)
            if mode in ("rossler", "dual", "morph"):
                self._reseed_fraction(self._rossler_pts, frac, *self._rossler_norm)

        # ── 2. ADAPTIVE TRAIL DECAY ──────────────────────────────────────
        # Quiet passages: long ghostly trails (near 0.99).
        # Loud drops: sharp punchy strokes (down to ~0.80).
        # key_stability (C4): stable key → longer trails; atonal → shorter, more volatile.
        stability_bonus = self._smooth_key_stability * 0.04  # 0 to +0.04
        decay = (self.cfg.trail_decay + stability_bonus) * (1.0 - self._smooth_energy * 0.20 * s)
        self._accum *= float(np.clip(decay, 0.78, 0.999))

        # ── 3. ATTRACTOR PARAMETERS ──────────────────────────────────────
        sigma, rho, beta, a, b, c = self._modulate_params()

        # ── 4. ENERGY-DRIVEN SIMULATION SPEED ───────────────────────────
        # Music energy speeds up the orbit — loud sections feel kinetic.
        dt_mult = 1.0 + self._smooth_energy * 1.5 * s
        effective_substep_dt = self._DT_SIM * dt_mult / self.cfg.substeps

        # ── 5. INTEGRATE PARTICLES ───────────────────────────────────────
        if mode in ("lorenz", "dual", "morph"):
            rk4_lorenz(
                self._lorenz_pts,
                sigma, rho, beta,
                effective_substep_dt,
                self.cfg.substeps,
            )
            self._fix_nans(self._lorenz_pts, self._lorenz_norm[0])

        if mode in ("rossler", "dual", "morph"):
            rk4_rossler(
                self._rossler_pts,
                a, b, c,
                effective_substep_dt,
                self.cfg.substeps,
            )
            self._fix_nans(self._rossler_pts, self._rossler_norm[0])

        # ── 6. BASS SCALE PULSE + BAR BREATHING (C5) ────────────────────
        # Sub-bass hits make the cloud breathe outward.
        # beat_position adds a subtle metronomic inhale/exhale per bar:
        # cloud contracts at beat 1, expands toward beat 4.
        bar_breath = 1.0 + math.cos(beat_position * 2.0 * math.pi) * 0.04 * s
        scale_target = (1.0 + self._smooth_sub_bass * 0.40 * s) * bar_breath
        self._scale_pulse = self._lerp(self._scale_pulse, float(scale_target), 0.30)

        # ── 7. PERCUSSION BRIGHTNESS PULSE ──────────────────────────────
        # Drum hits flare the particle brightness directly.
        bright_target = 1.0 + self._smooth_percussive * 2.0 * s
        self._brightness_pulse = self._lerp(
            self._brightness_pulse, float(bright_target), 0.45
        )

        # ── 8. BEAT SHOCKWAVE ────────────────────────────────────────────
        # Downbeat (C5): 1.8× camera kick, higher reseed base — bar 1 hits hardest.
        # onset_type (W4): transient onsets get an extra azimuth agitation;
        # harmonic onsets skip reseed (they're smooth notes, not drum hits).
        if is_beat:
            kick_scale = 1.8 if is_downbeat else 1.0
            kick_angle = self._smooth_percussive * 0.30 * s * kick_scale
            self._proj_az += kick_angle * (1.0 if (self.rng.random() > 0.5) else -1.0)
            reseed_base = 0.18 if is_downbeat else 0.12
            reseed_frac = float(np.clip(
                reseed_base + self._smooth_percussive * 0.28 * s, 0.05, 0.55
            ))
            # Harmonic onsets are smooth; skip the chaotic reseed
            if onset_type != "harmonic":
                if mode in ("lorenz", "dual", "morph"):
                    self._reseed_fraction(
                        self._lorenz_pts, reseed_frac, *self._lorenz_norm
                    )
                if mode in ("rossler", "dual", "morph"):
                    self._reseed_fraction(
                        self._rossler_pts, reseed_frac, *self._rossler_norm
                    )
            # Transient onsets (W4) add an extra azimuth micro-jolt
            if onset_type == "transient" and onset_sharpness > 0.6:
                self._proj_az += self.rng.normal(0.0, 0.06 * onset_sharpness * s)
        elif self._smooth_brilliance > 0.45:
            # Subtle hi-hat/cymbal sparkle
            reseed_frac = (self._smooth_brilliance - 0.45) * 0.08 * s
            if mode in ("lorenz", "dual", "morph"):
                self._reseed_fraction(
                    self._lorenz_pts, reseed_frac, *self._lorenz_norm
                )
            if mode in ("rossler", "dual", "morph"):
                self._reseed_fraction(
                    self._rossler_pts, reseed_frac, *self._rossler_norm
                )

        # ── 9. CAMERA ROTATION ───────────────────────────────────────────
        # Energy drives rotation speed.
        rot_speed = self.cfg.projection_speed * (1.0 + self._smooth_energy * 2.5 * s)
        self._proj_az += rot_speed * dt
        # Elevation: pitch_register (C3) maps the actual melody to vertical space.
        # Bass-heavy passages → level/low view (0.05); high soprano → looking up (0.60).
        # Previously this used harmonic_energy which is just amplitude, not register.
        el_target = 0.05 + self._smooth_pitch_register * 0.55
        self._proj_el = self._lerp(self._proj_el, float(el_target), 0.04)

        # ── 10. PITCH-DRIVEN HUE ─────────────────────────────────────────
        # Dominant chord note rotates the palette — the music's key has a color.
        # key_stability (C4) modulates how eagerly we chase pitch_hue:
        # in a stable key the palette tracks chord changes freely;
        # in chromatic/atonal passages it holds steady to avoid jitter.
        pitch_hue = float(frame_data.get("pitch_hue", 0.0))
        hue_lerp_rate = 0.04 + self._smooth_key_stability * 0.08
        self._hue_target = self._lerp(self._hue_target, pitch_hue, hue_lerp_rate)
        centroid = float(frame_data.get("spectral_centroid", 0.5))
        # Pitch_color_strength controls how boldly the chord note shifts hue
        self._hue_offset = (
            self._hue_target * self.cfg.pitch_color_strength
            + centroid * 0.008            # gentle spectral drift
            + self._smooth_brilliance * 0.04  # hi-freq shimmer
        ) % 1.0

        # ── 11. PROJECT AND SPLAT ─────────────────────────────────────────
        sm = float(self._scale_pulse)
        bm = float(self._brightness_pulse)

        if mode == "lorenz":
            self._project_and_splat(
                self._lorenz_pts, self._lorenz_norm, "lorenz", palette, H, W, sm, bm
            )
        elif mode == "rossler":
            self._project_and_splat(
                self._rossler_pts, self._rossler_norm, "rossler", palette, H, W, sm, bm
            )
        elif mode == "dual":
            self._project_and_splat(
                self._lorenz_pts, self._lorenz_norm, "lorenz", palette, H, W, sm, bm
            )
            self._project_and_splat(
                self._rossler_pts, self._rossler_norm, "rossler", palette, H, W, sm, bm
            )
        elif mode == "morph":
            blend_weight = float(np.clip(self._smooth_flatness, 0.0, 1.0))
            lc, ls = self._lorenz_norm
            rc, rs = self._rossler_norm
            lorenz_n = (self._lorenz_pts - lc) / ls
            rossler_n = (self._rossler_pts - rc) / rs
            N_pts = min(len(lorenz_n), len(rossler_n))
            morph_n = (
                (1.0 - blend_weight) * lorenz_n[:N_pts]
                + blend_weight * rossler_n[:N_pts]
            )
            rot = _rotation_matrix(self._proj_az, self._proj_el)
            pts_rot = morph_n @ rot.T
            half_side = min(W, H) / 2.0 * 0.85 * sm
            x_px = W / 2.0 + pts_rot[:, 0] * half_side
            y_px = H / 2.0 - pts_rot[:, 1] * half_side
            z_depth = np.clip(
                (pts_rot[:, 2] + 1.0) / 2.0, 0.0, 1.0
            ).astype(np.float32)
            rgb_l = self._compute_colors(z_depth, "lorenz", palette)
            rgb_r = self._compute_colors(z_depth, "rossler", palette)
            rgb = (
                (1.0 - blend_weight) * rgb_l + blend_weight * rgb_r
            ).astype(np.float32)
            weights = (
                self.cfg.particle_brightness * bm * (0.3 + 0.7 * z_depth)
            ).astype(np.float32)
            splat_soft(
                x_px.astype(np.float64),
                y_px.astype(np.float64),
                rgb[:, 0], rgb[:, 1], rgb[:, 2],
                weights,
                self._accum, H, W,
                float(self.cfg.splat_radius),
            )

    def get_raw_field(self) -> np.ndarray:
        """Return luminance of accumulation buffer (for BaseVisualizer compat)."""
        return np.clip(self._accum.mean(axis=2), 0.0, 1.0).astype(np.float32)

    def render_frame(
        self, frame_data: Dict[str, Any], frame_index: int
    ) -> np.ndarray:
        """Override: multi-scale bloom + ACES filmic tone-map + saturation boost.

        Multi-scale bloom: tight core glow + mid halo + wide corona, all
        beat-flash responsive.  ACES filmic curve replaces Reinhard for richer
        midtones and more natural highlight roll-off.  A luma-preserving
        saturation multiply (~+30%) makes neon colours pop.
        """
        self.update(frame_data)

        # ── Multi-scale bloom pyramid ──────────────────────────────────────
        # Three overlapping gaussian radii produce: glowing core → soft halo
        # → wide cinematic corona.  Beat flash expands the outer two layers.
        glow_sigma = float(self.cfg.glow_radius)
        flash = 1.0 + self._beat_flash * 0.7

        b1 = gaussian_filter(self._accum, sigma=[glow_sigma, glow_sigma, 0])
        b2 = gaussian_filter(
            self._accum,
            sigma=[glow_sigma * 3.0 * flash, glow_sigma * 3.0 * flash, 0],
        )
        b3 = gaussian_filter(
            self._accum,
            sigma=[glow_sigma * 8.0 * flash, glow_sigma * 8.0 * flash, 0],
        )
        composite = self._accum + 0.7 * b1 + 0.4 * b2 + 0.15 * b3

        # ── ACES filmic tone-map ───────────────────────────────────────────
        # Hill (2015) ACES approximation — richer than Reinhard at high
        # exposures, preserves more colour in the midtones.
        exposure = self.cfg.particle_brightness * (1.0 + self._beat_flash * 0.8)
        x = composite * exposure
        _a, _b, _c, _d, _e = 2.51, 0.03, 2.43, 0.59, 0.14
        mapped = np.clip(
            (x * (_a * x + _b)) / (x * (_c * x + _d) + _e), 0.0, 1.0
        ).astype(np.float32)

        # ── Luma-preserving saturation boost ──────────────────────────────
        # Pull colours away from neutral grey by 30% — makes the neon palette
        # vivid rather than pastel.
        lum = (
            0.2126 * mapped[..., 0]
            + 0.7152 * mapped[..., 1]
            + 0.0722 * mapped[..., 2]
        )[..., np.newaxis]
        mapped = np.clip(lum + 1.3 * (mapped - lum), 0.0, 1.0)

        out = (mapped * 255.0).astype(np.uint8)

        # Vignette relaxes on beats — the world opens up at the drop
        if self.cfg.vignette_strength > 0:
            vign_str = self.cfg.vignette_strength * max(0.0, 1.0 - self._beat_flash * 0.35)
            out = vignette(out, strength=float(vign_str))

        # Chromatic aberration spikes with percussion — transients feel sharp
        if self.cfg.aberration_enabled and self.cfg.aberration_offset > 0:
            ab_off = int(
                self.cfg.aberration_offset * (1.0 + self._smooth_percussive * 2.5)
            )
            out = chromatic_aberration(out, offset=ab_off)

        return out
