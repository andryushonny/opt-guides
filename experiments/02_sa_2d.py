import matplotlib.pyplot as plt

from targets.benchmark_functions import TARGETS
from targets.visualize import trajectory_overlay, convergence_plot
from solvers.sa_template import simulated_annealing

histories = {}

for name, target in TARGETS.items():
    if target.dim != 2: # due to trajectory_overlay limitations
        continue
    history = simulated_annealing(target, n_iter=250, 
                                  t_start=1.0, t_end=1e-3)
    trajectory_overlay(target, history.xs, accepted_mask=history.accepted)
    histories[name] = history.fs

    plt.savefig(f"results/02_{name}_trajectory.png")

convergence_plot(histories, optimum_f=None)

plt.savefig(f"results/02_convergence.png")

