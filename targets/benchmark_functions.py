# ABOUTME: Pre-written test functions for continuous black-box optimization.
# Every Target exposes the same interface so SA, scikit-optimize, and any other
# solver can consume them uniformly: a callable, search bounds, and known
# global optima for verification.
#
# Add new targets to TARGETS at the bottom. Use make_gaussian_mixture() to
# construct synthetic landscapes with a global optimum you control.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# ============================================================================
# TYPES
# ============================================================================


@dataclass(frozen=True)
class Target:
    """A black-box target function with known optima.

    fn        : maps an array of shape (D,) to a scalar.
    bounds    : list of (lo, hi) per dimension. Length == dim.
    optima_x  : list of global minimizers (each shape (D,)). May contain >1 entry
                for problems with multiple equal global minima (Himmelblau, Branin).
    optimum_f : the global minimum value (same value at every entry of optima_x).
    """

    name: str
    fn: Callable[[np.ndarray], float]
    bounds: list[tuple[float, float]]
    optima_x: list[np.ndarray]
    optimum_f: float

    @property
    def dim(self) -> int:
        return len(self.bounds)

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(self.fn(x))


# ============================================================================
# 2D BENCHMARKS
# ============================================================================


def _rosenbrock(x: np.ndarray) -> float:
    # Curved valley along y = x^2; gradient method's nightmare. Min at (1,1,...).
    x = np.asarray(x)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def _rastrigin(x: np.ndarray) -> float:
    # Regular grid of local minima inside a parabolic envelope. Min at origin.
    x = np.asarray(x)
    return float(10.0 * len(x) + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


def _himmelblau(x: np.ndarray) -> float:
    # Four equal global minima at f=0. Classic motivator for multi-start.
    return float((x[0] ** 2 + x[1] - 11.0) ** 2 + (x[0] + x[1] ** 2 - 7.0) ** 2)


def _ackley(x: np.ndarray) -> float:
    # Funnel + many shallow local minima. Min at origin (=0).
    x = np.asarray(x)
    n = len(x)
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(2.0 * np.pi * x))
    return float(
        -20.0 * np.exp(-0.2 * np.sqrt(s1 / n))
        - np.exp(s2 / n)
        + 20.0
        + np.e
    )


def _branin(x: np.ndarray) -> float:
    # The de-facto Bayesian-optimization test function. 3 equal global minima.
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    return float(
        a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2
        + s * (1.0 - t) * np.cos(x[0])
        + s
    )


# ============================================================================
# N-D BENCHMARKS
# ============================================================================

# Hartmann-3 constants (Dixon & Szegő 1978; reproduced in every BO paper).
_H3_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_H3_A = np.array(
    [
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
    ]
)
_H3_P = np.array(
    [
        [0.3689, 0.1170, 0.2673],
        [0.4699, 0.4387, 0.7470],
        [0.1091, 0.8732, 0.5547],
        [0.0381, 0.5743, 0.8828],
    ]
)


def _hartmann3(x: np.ndarray) -> float:
    # 3D, single global minimum at f ≈ -3.86278.
    x = np.asarray(x)
    inner = np.sum(_H3_A * (x[None, :] - _H3_P) ** 2, axis=1)
    return float(-np.sum(_H3_ALPHA * np.exp(-inner)))


# ============================================================================
# SYNTHETIC: GAUSSIAN MIXTURE LANDSCAPE
# ============================================================================


def make_gaussian_mixture(
    centers: np.ndarray,
    depths: np.ndarray,
    widths: np.ndarray,
    bounds: list[tuple[float, float]],
    name: str = "gaussian_mixture",
) -> Target:
    """Build a landscape as a sum of isotropic Gaussian wells.

    f(x) = sum_k depth_k * exp(-||x - center_k||^2 / (2 * width_k^2))

    depth_k is signed: negative = well (attractor under minimization),
    positive = bump (repeller). The global minimum is taken to be the centre
    of the deepest well, valid when wells are well-separated relative to
    their widths. If wells overlap, optima_x is approximate.

    centers : (K, D) array of well/bump centres.
    depths  : (K,) signed amplitudes. Negative for wells (what you usually want).
    widths  : (K,) Gaussian sigmas.
    bounds  : per-dimension search bounds (length D).
    """
    centers = np.asarray(centers, dtype=float)
    depths = np.asarray(depths, dtype=float)
    widths = np.asarray(widths, dtype=float)
    K, D = centers.shape
    assert depths.shape == (K,) and widths.shape == (K,)
    assert len(bounds) == D

    def fn(x: np.ndarray) -> float:
        sq = np.sum((centers - x[None, :]) ** 2, axis=1)
        return float(np.sum(depths * np.exp(-sq / (2.0 * widths**2))))

    deepest = int(np.argmin(depths))
    approx_optimum_x = centers[deepest].copy()
    approx_optimum_f = fn(approx_optimum_x)

    return Target(
        name=name,
        fn=fn,
        bounds=list(bounds),
        optima_x=[approx_optimum_x],
        optimum_f=approx_optimum_f,
    )


# ============================================================================
# REGISTRY
# ============================================================================

TARGETS: dict[str, Target] = {
    "rosenbrock_2d": Target(
        name="rosenbrock_2d",
        fn=_rosenbrock,
        bounds=[(-2.0, 2.0), (-1.0, 3.0)],
        optima_x=[np.array([1.0, 1.0])],
        optimum_f=0.0,
    ),
    "rastrigin_2d": Target(
        name="rastrigin_2d",
        fn=_rastrigin,
        bounds=[(-5.12, 5.12), (-5.12, 5.12)],
        optima_x=[np.array([0.0, 0.0])],
        optimum_f=0.0,
    ),
    "himmelblau": Target(
        name="himmelblau",
        fn=_himmelblau,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        optima_x=[
            np.array([3.0, 2.0]),
            np.array([-2.805118, 3.131312]),
            np.array([-3.779310, -3.283186]),
            np.array([3.584428, -1.848126]),
        ],
        optimum_f=0.0,
    ),
    "ackley_2d": Target(
        name="ackley_2d",
        fn=_ackley,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        optima_x=[np.array([0.0, 0.0])],
        optimum_f=0.0,
    ),
    "branin": Target(
        name="branin",
        fn=_branin,
        bounds=[(-5.0, 10.0), (0.0, 15.0)],
        optima_x=[
            np.array([-np.pi, 12.275]),
            np.array([np.pi, 2.275]),
            np.array([9.42478, 2.475]),
        ],
        optimum_f=0.397887,
    ),
    "rosenbrock_5d": Target(
        name="rosenbrock_5d",
        fn=_rosenbrock,
        bounds=[(-2.0, 2.0)] * 5,
        optima_x=[np.ones(5)],
        optimum_f=0.0,
    ),
    "rastrigin_5d": Target(
        name="rastrigin_5d",
        fn=_rastrigin,
        bounds=[(-5.12, 5.12)] * 5,
        optima_x=[np.zeros(5)],
        optimum_f=0.0,
    ),
    "hartmann_3d": Target(
        name="hartmann_3d",
        fn=_hartmann3,
        bounds=[(0.0, 1.0)] * 3,
        optima_x=[np.array([0.114614, 0.555649, 0.852547])],
        optimum_f=-3.86278,
    ),
}
