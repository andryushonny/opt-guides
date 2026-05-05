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

    xs           : (T+1, D) — every visited point including the start.
    fs           : (T+1,)   — f at each visited point.
    accepted     : (T,) bool — whether each proposal was accepted.
    temperatures : (T,)   — temperature used when proposing each move.
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

    # TODO: choose x0 sensibly if None — uniform in bounds is a fine default.
    if x0 is None:
        raise NotImplementedError("TODO: sample x0 uniformly within target.bounds")
    x = np.asarray(x0, dtype=float).copy()

    # TODO: compute the geometric cooling factor alpha so that
    # t_start * alpha**n_iter == t_end. Watch for n_iter == 0.
    alpha = ...  # noqa: F841

    # Pre-allocate history arrays of the right shape.
    xs = np.zeros((n_iter + 1, D))
    fs = np.zeros(n_iter + 1)
    accepted = np.zeros(n_iter, dtype=bool)
    temperatures = np.zeros(n_iter)

    xs[0] = x
    fs[0] = target(x)
    best_x = x.copy()
    best_f = fs[0]

    T = t_start
    for k in range(n_iter):
        # TODO: propose a candidate x_new by perturbing x with a Gaussian step
        # of std = step_size, then clipping to [lo, hi].
        x_new = ...  # noqa: F841

        # TODO: evaluate f_new = target(x_new).
        f_new = ...  # noqa: F841

        # TODO: Metropolis acceptance.
        # delta = f_new - target_f_at_x
        # if delta < 0: accept
        # else: accept with probability exp(-delta / T)
        accept = ...  # noqa: F841

        # TODO: if accepted, update x and the running f. Update best_x/best_f
        # whenever the accepted point is better than best_f.

        # Bookkeeping (do not modify — visualize.py depends on these shapes).
        xs[k + 1] = x
        fs[k + 1] = target(x)        # cheap; cache if you care about perf
        accepted[k] = bool(accept)
        temperatures[k] = T

        # TODO: cool. T *= alpha
        ...

    return SAHistory(
        xs=xs,
        fs=fs,
        accepted=accepted,
        temperatures=temperatures,
        best_x=best_x,
        best_f=best_f,
    )
