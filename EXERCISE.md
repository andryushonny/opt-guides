# Optimization Exercise — Simulated Annealing & Bayesian Optimization

## Proposal (informal note)

Four stages. Each stage produces one runnable `.py` file in `experiments/`
and a small set of plots. There is no autograder — your evidence is your
plots and your written answers to the discussion prompts.

Pace this however suits you. The whole arc is sized to a focused weekend
or two; if you go deep on diagnostics it's longer. Don't optimize for
speed — optimize for "I understand why this curve looks like that."

You start with stage 1 already done as a worked example
(`experiments/01_visualize.py`) so you can see the project's cell-style
format. Stages 2-4 you write yourself, following the pointers below.

## Setup

```bash
cd /path/to/opt-guides
uv venv
source .venv/bin/activate
uv pip install -e .       # installs deps and makes `targets` / `solvers` importable
```

Open any `experiments/*.py` in VS Code / Cursor / PyCharm. The `# %%`
markers turn the file into a notebook: Shift+Enter runs the current cell
inline. Plain `python experiments/01_visualize.py` also works.

## Files you should know

| Path | What it is | Touch it? |
|---|---|---|
| `targets/benchmark_functions.py` | Test-function registry, ground-truth optima. | Read. |
| `targets/visualize.py` | Plot helpers — contour, surface, trajectory, convergence. | Use, don't edit. |
| `targets/__init__.py` | Re-exports. | Skip. |
| `solvers/sa_template.py` | **Your job — fill in the TODOs.** | **Edit.** |
| `experiments/01_visualize.py` | Worked example showing the cell format. | Read, then run. |
| `experiments/02_sa_2d.py` | You write this. | **Create.** |
| `experiments/03_sa_nd.py` | You write this. | **Create.** |
| `experiments/04_skopt_compare.py` | You write this. | **Create.** |

## Stage 1 — see the landscape *(already done)*

Run `experiments/01_visualize.py`. Read it. Notice:

- Every `Target` has the same interface: `target(x)` evaluates it,
  `target.bounds` and `target.optima_x` give you ground truth.
- The cell format: `# %%` separates cells. Shift+Enter runs them.

**Discussion prompt** (no code, just think): which of the 5 visualized
2D targets do you expect SA to crush, which to struggle with, and why?
Write your prediction in a comment in `experiments/02_sa_2d.py` before
you run anything. Compare to what actually happens.

## Stage 2 — hand-roll SA on 2D targets

**Implement `solvers/sa_template.py`.** Every TODO must go away. The
algorithm is 5 lines of math and ~30 of Python:

1. propose `x' = x + N(0, step_size)`, clip to bounds
2. `Δ = f(x') − f(x)`
3. accept if `Δ < 0`, else accept with probability `exp(−Δ / T)`
4. cool `T` geometrically so `T(n_iter) == t_end`
5. track `best_x, best_f` ever seen

Then write `experiments/02_sa_2d.py`. It should:

- Run your SA on at least Rosenbrock, Rastrigin, and Himmelblau.
- Plot the **trajectory overlay** for each (use
  `targets.visualize.trajectory_overlay`).
- Plot a **convergence curve** comparing the three runs (use
  `convergence_plot`). The y-axis is best-f-so-far minus the known
  optimum, log-scale.
- Try a few `(t_start, t_end, n_iter, step_size)` combinations. Keep
  the ones that show interesting behavior — a plot where the chain
  obviously *doesn't* converge is just as instructive as one that does.

**Discussion prompt** (write 3-5 sentences in a markdown cell at the
end of `02_sa_2d.py`): for one of the three problems, does your chain
get stuck in a local minimum? At what temperature? What happens if you
double `step_size`? Halve it?

**Hint on Himmelblau**: with 4 equal global minima, a single SA chain
will land at one of them — which one depends on the seed. This is the
natural lead-in to stage 4 (multi-start). Make a note when you see it.

## Stage 3 — go to higher dimensions

Write `experiments/03_sa_nd.py`. Run your SA on `rosenbrock_5d`,
`rastrigin_5d`, and `hartmann_3d`. Same convergence-curve idiom as
stage 2 (you can't plot 5D trajectories — convergence is the diagnostic).

You'll likely see SA struggle more than in 2D, especially on
`rosenbrock_5d` (the curved valley extends across all 5 dims). Push
`n_iter` until convergence flattens, then push again. Look at the
runtime cost — does it scale linearly with `n_iter`? With `D`?

**Discussion prompt**: for fixed budget (say 20,000 evaluations), how
does the gap to the known optimum scale as you go from D=2 to D=5?
Write your numbers in a markdown cell. This is your first taste of the
curse of dimensionality.

## Stage 4 — Bayesian optimization & multi-start

Write `experiments/04_skopt_compare.py`. Use scikit-optimize's
[`gp_minimize`](https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html).

Two comparisons:

**A) BO vs. SA at small budget.** Pick `branin` (a BO classic) and
`rastrigin_2d`. Run both methods with the *same* total evaluation
budget — try 50 and 100. Plot convergence curves on one axis: SA, BO,
and a "random search" baseline (just sample uniformly from bounds).
Branin should favor BO; Rastrigin should favor SA. Why?

**B) Parallel multi-start.** For SA, run N independent chains from
random starts (use different RNG seeds), take the best. Try N ∈ {1, 4,
16}. Plot: best-of-N convergence vs. evaluation count
(*total* evaluations across all chains — that's the fair x-axis).
Compare against BO with `n_initial_points = N`.

**Discussion prompt**: when does multi-start help most? What does
"parallel" buy you that "longer single chain" doesn't? When would you
pick BO over multi-start SA in practice?

## A short aside: ML loss landscapes

You might wonder: can I just use these tools to tune ML hyperparameters?
You can — but be aware of three things real ML landscapes have that the
benchmarks here don't:

1. **They are very high-dimensional.** A neural net has millions of
   parameters; even hyperparameter spaces are easily 10-30 dimensional.
   Your SA from stage 3 will not scale; BO scales to maybe ~20 dims
   before its GP surrogate breaks down.
2. **They are noisy.** Each evaluation is a stochastic training run.
   Same hyperparams + different seed = different validation loss. SA
   and BO both *can* handle this, but you need explicit averaging or
   acquisition functions that model noise (`gp_minimize` has
   `noise=...`).
3. **They are expensive.** One evaluation can take minutes to days.
   This is exactly the regime BO was invented for, but in practice
   people use specialized libraries (Optuna, Ray Tune, Ax) that combine
   BO with **early stopping** (ASHA, Hyperband) — kill bad runs early
   instead of waiting for them to finish. Pure BO leaves a lot on the
   table here.

Also: published "loss landscape" pictures of trained nets (Li et al.,
2018, *Visualizing the Loss Landscape of Neural Nets*) look
deceptively smooth — those are 2D *projections* of a million-dim
landscape. The local geometry near a trained net's minimum is benign;
the global geometry is something we can't picture.

So: 2D contour plots are pedagogy, not a faithful preview of ML
optimization. The intuitions from this exercise (acceptance probability,
exploration-exploitation, multi-start) carry over. The visual
intuitions mostly don't.

## Optional rabbit holes

If you finish and want more:

- **Adaptive step size.** Make `step_size` shrink as `T` cools, or as
  acceptance rate drops below a target. Compare to fixed step.
- **Restart-from-best.** When a chain stagnates, jump back to the best
  point seen and re-heat. Does it help on Himmelblau?
- **`forest_minimize`.** Same skopt API, random-forest surrogate
  instead of GP. When does it beat GP?
- **CMA-ES.** A serious continuous optimizer (`pip install cma`).
  Drop it into your benchmark — it will likely beat both SA and BO on
  Rosenbrock-N. Ask yourself why.

## Working with git

This repo is git-tracked. You don't have to commit, but I recommend it:
each time you finish a stage, commit. Small commits are better than big
ones. Suggested workflow:

```bash
git status                           # see what changed
git diff                             # see the diff
git add experiments/02_sa_2d.py solvers/sa_template.py
git commit -m "stage 2: SA on 2D targets"
```

If you've never used git before, this is a good moment to start. The
free side of <https://learngitbranching.js.org/> is the fastest way to
build intuition; for everyday workflow, `git status`, `git diff`,
`git add <file>`, `git commit -m "..."`, `git log --oneline` is 90% of
what you need.
