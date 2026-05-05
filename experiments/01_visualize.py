# ABOUTME: Stage 1 — see the landscapes you'll be optimizing on.
# This script is fully worked so you can see the cell-style .py format
# (run cell-by-cell in VS Code / Cursor / PyCharm with Shift+Enter, or run
# the whole file with `python experiments/01_visualize.py`).
#
# Stages 2-4 you write yourself. EXERCISE.md tells you what each should do.

# %%
import matplotlib.pyplot as plt

from targets import TARGETS
from targets.visualize import contour_plot, surface_plot

# %% [markdown]
# ## All 2D targets at a glance
# Each one has a known global optimum (red star). Note the very different
# shapes — these are the kinds of landscapes we'll throw optimizers at.

# %%
two_d_names = [name for name, t in TARGETS.items() if t.dim == 2]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, name in zip(axes.flat, two_d_names):
    contour_plot(TARGETS[name], ax=ax)
for ax in axes.flat[len(two_d_names):]:
    ax.set_visible(False)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Surface plot of a single target
# Useful for getting an intuition about depth/curvature. Try changing the
# name to "rastrigin_2d" to see what makes it hard for gradient methods.

# %%
fig, ax = surface_plot(TARGETS["rosenbrock_2d"])
plt.show()

# %% [markdown]
# ## Designing your own landscape with make_gaussian_mixture
# The wells are at known centres with known depths, so you have ground
# truth. Use this constructor in stage 2 if you want to invent harder
# problems for your SA.

# %%
import numpy as np

from targets import make_gaussian_mixture

custom = make_gaussian_mixture(
    centers=np.array([[-2.0, -2.0], [2.0, 2.0], [0.0, 3.5], [3.0, -3.0]]),
    depths=np.array([-1.0, -3.0, -1.5, -2.5]),    # deepest at (2,2)
    widths=np.array([0.8, 0.6, 0.5, 1.0]),
    bounds=[(-5.0, 5.0), (-5.0, 5.0)],
    name="custom_4_wells",
)
print(f"Designed optimum: x={custom.optima_x[0]}  f={custom.optimum_f:.4f}")
fig, ax = contour_plot(custom)
plt.show()
