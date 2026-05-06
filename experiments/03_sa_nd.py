import matplotlib.pyplot as plt

import time 

from targets.benchmark_functions import TARGETS
from targets.visualize import convergence_plot
from solvers.sa_template import simulated_annealing

target_names = ['rosenbrock_5d', 
                'rastrigin_5d',
                'hartmann_3d']

histories = {}

CURR_TARGETS = {k: TARGETS[k] for k in target_names}

# %%
n_iters = [25, 50, 100, 200, 250, 500, 1000, 2500, 5000]

elapsed_all = {t: [] for t in target_names}

fig, ax = plt.subplots()

for n_iter in n_iters:
    for name, target in CURR_TARGETS.items():
        start = time.perf_counter()
        history = simulated_annealing(target, n_iter=n_iter, 
                                    t_start=2.0, t_end=1e-4)
        histories[name] = history.fs

        elapsed = time.perf_counter() - start
        elapsed_all[name].append(elapsed)

    convergence_plot(histories, optimum_f=None)

    plt.savefig(f"results/03_convergence_niter={n_iter}.png")

for name, target in CURR_TARGETS.items():
    ax.plot(n_iters, elapsed_all[name], label=name)

ax.set_xlabel("n_iter")
ax.set_ylabel("Runtime, s")
ax.set_xlim(0, max(n_iters))

ax.legend()

fig.savefig(f"results/03_runtime.png")

# %% 

n_iter = 20000

gaps = {target:0 for target in CURR_TARGETS.keys()}

for name, target in CURR_TARGETS.items():
    start = time.perf_counter()
    history = simulated_annealing(target, n_iter=n_iter, 
                                t_start=2.0, t_end=1e-4)
    histories[name] = history.fs
    gaps[name] = history.best_f - target.optimum_f

print(gaps)
    
convergence_plot(histories, optimum_f=None)

