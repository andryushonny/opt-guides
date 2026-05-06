# ABOUTME: Stub for the student to implement Simulated Annealing.
# Fill in every TODO. Keep the function signature and the returned History
# shape, because targets/visualize.py and the experiment scripts assume them.
#
# Recommended reading before you start: 
#   https://en.wikipedia.org/wiki/Simulated_annealing
#   Numerical Recipes, ch. 10.9 — concise, classic write-up.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from targets.benchmark_functions import Target


# ============================================================================
# RESULT TYPE
# ============================================================================


@dataclass
class SAHistory:
    """What every SA run returns. The experiment scripts and visualize.py
    expect exactly these fields.

    xs           : (N+1, D) — every visited point including the start.
    fs           : (N+1,)   — f at each visited point.
    accepted     : (N+1,) bool — whether each proposal was accepted.
    temperatures : (N,)   — temperature used when proposing each move.
    best_x       : (D,)   — argmin over the run.
    best_f       : float  — min over the run.
    """

    xs: np.ndarray
    fs: np.ndarray
    accepted: np.ndarray
    temperatures: np.ndarray
    best_x: np.ndarray
    best_f: float


# ============================================================================
# SIMULATED ANNEALING
# ============================================================================


def simulated_annealing(
    target: Target,
    x0: np.ndarray | None = None,
    n_iter: int = 2000,
    t_start: float = 1.0,
    t_end: float = 1e-3,
    step_size: float = 0.5,
    rng: np.random.Generator | None = None,
) -> SAHistory:
    """Minimize target.fn with Metropolis simulated annealing.

    The algorithm in 5 lines:
      1. propose x' = x + Gaussian(step_size)
      2. clip x' to bounds
      3. if f(x') < f(x): accept
         else: accept with probability exp(-(f(x') - f(x)) / T)
      4. cool T (geometric: T_{k+1} = alpha * T_k, with alpha derived from
         t_start, t_end, n_iter so that T(n_iter) == t_end exactly)
      5. track best ever seen

    Returns an SAHistory with the full trajectory for diagnostics.
    """
    if rng is None:
        rng = np.random.default_rng()

    bounds = np.asarray(target.bounds)         # shape (D, 2)
    lo, hi = bounds[:, 0], bounds[:, 1]
    D = target.dim

    if x0 is None:
        x0 = rng.uniform(lo, hi)

    x = np.asarray(x0, dtype=float).copy()

    # alpha = decay rate
    alpha = (t_end / t_start) ** (1 / n_iter) if n_iter > 0 else 1

    # Pre-allocate history arrays of the right shape.
    xs = np.zeros((n_iter + 1, D))
    fs = np.zeros(n_iter + 1)
    accepted = np.zeros(n_iter + 1, dtype=bool)
    accepted[0] = True
    temperatures = np.zeros(n_iter)

    xs[0] = x
    fs[0] = target(x)
    best_x = x.copy()
    best_f = fs[0]

    T = t_start
    for k in range(n_iter):
        x_new = np.clip(x + rng.normal(loc=0, scale=step_size, size=D), lo, hi)

        f_new = target(x_new)

        delta = f_new - fs[k]
        prob = np.exp(np.clip(-delta / T, -100, 0))
        
        accept = rng.random() < prob if prob < 1 else True
        
        if accept: 
            x = x_new
            if f_new <= best_f:
                best_x = x.copy()
                best_f = f_new

        xs[k + 1] = x
        fs[k + 1] = target(x)       
        accepted[k + 1] = accept
        temperatures[k] = T

        T *= alpha

    return SAHistory(
        xs=xs,
        fs=fs,
        accepted=accepted,
        temperatures=temperatures,
        best_x=best_x,
        best_f=best_f,
    )
