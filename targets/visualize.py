# ABOUTME: Plot helpers for 2D Targets — contour, surface, trajectory overlay,
# and convergence curves. Imported by the experiment scripts so every plot in
# the project has a consistent look.
#
# All functions accept an optional matplotlib Axes; if None, a new figure is
# made and returned alongside the axes.

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .benchmark_functions import Target


# ============================================================================
# 2D LANDSCAPE PLOTS
# ============================================================================


def _grid(target: Target, n: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if target.dim != 2:
        raise ValueError(f"contour/surface plots are 2D-only (got dim={target.dim})")
    (x_lo, x_hi), (y_lo, y_hi) = target.bounds
    xs = np.linspace(x_lo, x_hi, n)
    ys = np.linspace(y_lo, y_hi, n)
    X, Y = np.meshgrid(xs, ys)
    Z = np.array([[target(np.array([x, y])) for x in xs] for y in ys])
    return X, Y, Z


def contour_plot(
    target: Target, ax: Axes | None = None, n: int = 200, levels: int = 30
) -> tuple[Figure, Axes]:
    """Filled contour of the target with each global optimum marked."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    X, Y, Z = _grid(target, n=n)
    cs = ax.contourf(X, Y, Z, levels=levels, cmap="viridis")
    ax.contour(X, Y, Z, levels=levels, colors="white", alpha=0.25, linewidths=0.5)
    fig.colorbar(cs, ax=ax, label="f(x)")
    for opt in target.optima_x:
        ax.plot(opt[0], opt[1], marker="*", color="red", markersize=14,
                markeredgecolor="white", markeredgewidth=1.0, zorder=5)
    ax.set_xlabel("x_0")
    ax.set_ylabel("x_1")
    ax.set_title(f"{target.name}  (optimum f={target.optimum_f:.4g})")
    return fig, ax


def surface_plot(target: Target, n: int = 80) -> tuple[Figure, Axes]:
    """3D surface plot. Returns its own figure (3D axes can't be passed in cleanly)."""
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    X, Y, Z = _grid(target, n=n)
    ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel("x_0")
    ax.set_ylabel("x_1")
    ax.set_zlabel("f(x)")
    ax.set_title(target.name)
    return fig, ax


# ============================================================================
# RUN DIAGNOSTICS
# ============================================================================


def trajectory_overlay(
    target: Target,
    xs: np.ndarray,
    ax: Axes | None = None,
    accepted_mask: np.ndarray | None = None,
) -> tuple[Figure, Axes]:
    """Draw a contour and overlay an optimizer's trajectory.

    xs            : array of shape (T, 2). Sequence of visited points.
    accepted_mask : optional bool array of shape (T,). If given, accepted moves
                    drawn solid, rejected as faint dots. Useful for SA debugging.
    """
    fig, ax = contour_plot(target, ax=ax)
    xs = np.asarray(xs)
    if accepted_mask is None:
        ax.plot(xs[:, 0], xs[:, 1], color="white", linewidth=1.0, alpha=0.7)
        ax.scatter(xs[:, 0], xs[:, 1], c=np.arange(len(xs)), cmap="autumn",
                   s=10, zorder=4)
    else:
        accepted = xs[accepted_mask]
        rejected = xs[~accepted_mask]
        ax.plot(accepted[:, 0], accepted[:, 1], color="white",
                linewidth=1.0, alpha=0.8)
        ax.scatter(rejected[:, 0], rejected[:, 1], color="grey",
                   s=4, alpha=0.4, zorder=3)
        ax.scatter(accepted[:, 0], accepted[:, 1],
                   c=np.arange(len(accepted)), cmap="autumn", s=12, zorder=4)
    ax.scatter(xs[0, 0], xs[0, 1], marker="o", color="cyan",
               s=80, edgecolor="black", zorder=6, label="start")
    ax.scatter(xs[-1, 0], xs[-1, 1], marker="X", color="magenta",
               s=100, edgecolor="black", zorder=6, label="end")
    ax.legend(loc="upper right")
    return fig, ax


def convergence_plot(
    histories: dict[str, np.ndarray],
    optimum_f: float | None = None,
    ax: Axes | None = None,
    log_y: bool = True,
) -> tuple[Figure, Axes]:
    """Best-so-far (cumulative min) vs. evaluation count, one curve per method.

    histories : {label: array of length T} where each entry is f at that eval.
                The function takes its own cumulative min, so pass raw f values.
    optimum_f : if given, drawn as a horizontal dashed line.
    log_y     : log-scale the y axis on (best - optimum_f) for nicer reading.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure
    for label, fs in histories.items():
        fs = np.asarray(fs)
        best = np.minimum.accumulate(fs)
        if log_y and optimum_f is not None:
            ax.semilogy(np.arange(1, len(best) + 1), best - optimum_f, label=label)
            ax.set_ylabel("best f − f*")
        else:
            ax.plot(np.arange(1, len(best) + 1), best, label=label)
            ax.set_ylabel("best f so far")
    if optimum_f is not None and not (log_y):
        ax.axhline(optimum_f, color="black", linestyle="--", alpha=0.5,
                   label=f"f* = {optimum_f:.4g}")
    ax.set_xlabel("function evaluations")
    ax.set_title("Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig, ax
