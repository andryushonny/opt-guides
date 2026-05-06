from skopt import gp_minimize

import time
import numpy as np
import matplotlib.pyplot as plt

from targets.benchmark_functions import TARGETS
from targets.visualize import convergence_plot
from solvers.sa_template import simulated_annealing

target_names = ['branin',
                'rastrigin_2d']

histories, histories_skopt, histories_rnd = {}, {}, {}

CURR_TARGETS = {k: TARGETS[k] for k in target_names}

n_iter = 50

def random_search(target, n_calls, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    lo = [b[0] for b in target.bounds]
    hi = [b[1] for b in target.bounds]
    fs = []
    for _ in range(n_calls):
        x = rng.uniform(lo, hi)
        fs.append(target(x))
    return np.array(fs)

for name, target in CURR_TARGETS.items():
    # Simulated Annealing
    sa_history = simulated_annealing(target, n_iter=n_iter, 
                                    t_start=1.0, t_end=1e-4)
    histories[name] = sa_history.fs

    # Scikit Optimize
    res = gp_minimize(target.fn, target.bounds, n_calls=n_iter, random_state=11)
    histories_skopt[name] = res.func_vals

    # Random search
    histories_rnd[name] = random_search(target, n_calls=n_iter, rng=np.random.default_rng(11))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for i, (name, target) in enumerate(CURR_TARGETS.items()):
    combined = {
        "Simulated Annealing": histories[name],
        "Scikit Optimize": histories_skopt[name],
        "Random Search": histories_rnd[name],
    }
    convergence_plot(combined, ax=axes[i])
    axes[i].set_title(name)

#fig, ax = plt.subplots(3, 1, figsize=(8, 10))
#convergence_plot(histories, ax=ax[0])
#convergence_plot(histories_skopt, ax=ax[1])
#convergence_plot(histories_rnd, ax=ax[2])

#ax[0].set_title("Simulated Annealing")
#ax[1].set_title("Scikit Optimize")
#ax[2].set_title("Random Search")

plt.tight_layout()
plt.savefig(f"results/04_convergence_comparison.png")

plt.show()

# %%
target_names = ['rastrigin_2d']

CURR_TARGETS = {k: TARGETS[k] for k in target_names}

n_iter = 64

N = [1, 4, 16] # number of parallel evaluations

histories = {}

start = time.perf_counter()

for nn in N:
    histories_temp = []    
    for i in range(nn):
        for name, target in CURR_TARGETS.items():
            sa_history = simulated_annealing(target, n_iter=n_iter, 
                                            t_start=1.0, t_end=1e-4, 
                                            rng=np.random.default_rng(seed=np.random.randint(0, 100)))
        histories_temp.append(sa_history.fs)
    histories[nn] = np.min(np.stack(histories_temp), axis=0)

elapsed = time.perf_counter() - start

print(f"Simulated Annealing took {elapsed:.2f} seconds in total.")

convergence_plot(histories)

plt.savefig(f"results/04_convergence_parallel.png")

histories = {}

start = time.perf_counter()

for nn in N:
    for name, target in CURR_TARGETS.items():
        res = gp_minimize(target.fn, target.bounds, n_initial_points=nn, n_calls=n_iter, 
                            random_state=np.random.randint(0, 100))
    histories[nn] = res.func_vals

convergence_plot(histories)

elapsed = time.perf_counter() - start

print(f"Scikit Optimize took {elapsed:.2f} seconds in total.")

plt.savefig(f"results/04_convergence_skopt.png")

plt.show()    

