# opt-guides

A hands-on exercise on continuous black-box optimization: write your own Simulated Annealing, then compare it against scikit-optimize's Bayesian optimization on a set of pre-built test functions.

## Start here

1. Skim this README.
2. Open and read [`EXERCISE.md`](EXERCISE.md) — it has the actual task,
   the four stages, and pointers to every file.
3. Set up the environment (below), run
   `python experiments/01_visualize.py` to confirm everything works,
   then start on stage 2.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency
management. If you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then from the repo root:

```bash
uv venv                         # creates .venv/
source .venv/bin/activate
uv pip install -e .             # installs deps + makes the project importable
```

To verify:

```bash
python experiments/01_visualize.py
```

You should see a grid of contour plots pop up.

## Project layout

```
opt-guides/
├── EXERCISE.md                       — the task: stages 1-4, hints, prompts
├── targets/
│   ├── benchmark_functions.py        — Rosenbrock, Rastrigin, Himmelblau,
│   │                                   Ackley, Branin, Hartmann-3, plus
│   │                                   make_gaussian_mixture for synthetic
│   │                                   landscapes you design yourself.
│   └── visualize.py                  — contour, surface, trajectory, and
│                                       convergence plot helpers.
├── solvers/
│   └── sa_template.py                — stub for your SA implementation.
├── experiments/
│   ├── 01_visualize.py               — worked example (read this first)
│   ├── 02_sa_2d.py                   — you write this
│   ├── 03_sa_nd.py                   — you write this
│   └── 04_skopt_compare.py           — you write this
└── pyproject.toml
```

## On the `.py` cell-style notebook format

The `experiments/*.py` files use `# %%` cell markers. In **VS Code**, **Cursor**, **PyCharm**, and most other modern editors, this turns the file into an interactive notebook: Shift+Enter runs the current cell inline (output appears beside the code). Plain `python file.py` also works for end-to-end runs.

This format is preferred over `.ipynb` because it diffs cleanly in git, edits cleanly in any editor, and doesn't carry around cached outputs. If you'd like a richer reactive notebook experience, look at [marimo](https://marimo.io/) — also `.py`-stored, but with reactive cell re-execution. Optional, not required.

## What you'll learn

- How simulated annealing works mechanically — proposals, acceptance, cooling — by writing it yourself.
- How to read optimizer diagnostics: trajectory overlays, convergence curves, acceptance rates.
- How Bayesian optimization compares: when its surrogate-model approach wins (smooth, expensive evaluations), when it loses (rugged, multi-modal landscapes).
- How parallel multi-start changes the picture, and why "best of N short runs" is sometimes better than "one long run."

## Prerequisites

NumPy basics, basic Python. No prior optimization experience required -- that's the point.
