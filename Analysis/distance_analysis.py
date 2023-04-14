import os

import ga_vqc as gav
import matplotlib.pyplot as plt
import numpy as np
import json

n_qubits = 3
n_moments = 4
gates_dict = {"I": (1, 0), "RX": (1, 1), "RY": (1, 1), "RZ": (1, 1), "PhaseShift": (1, 1), "CNOT": (2, 0)}
gates_probs = [0.15, 0.15, 0.15, 0.15, 0.15, 0.25]
genepool = gav.Genepool(gates_dict, gates_probs)
rng_seed = 4

dir = '/Users/tsievert/Downloads/reg_run_results'


dict_arr = []
for filename in os.listdir(dir):
    if filename[0] != '0':
        continue
    with open(os.path.join(dir, filename), 'r') as f: # open in readonly mode
        dict_arr.append(json.load(f))

best_ansatz_dicts = []
for moment_dict in dict_arr[0]["best_ansatz"]:
    best_ansatz_dicts.append({})
    for k, v in moment_dict.items():
        best_ansatz_dicts[-1][int(k)] = v

best_ansatz = gav.Individual(n_qubits, n_moments, genepool, rng_seed, ansatz_dicts=best_ansatz_dicts)

full_pop_ansatz = []
full_pop_auroc = []
full_pop_fitness = []
avg_pop_fitness = []
std_dev_pop_fitness = []
avg_pop_auroc = []

for results_dict in dict_arr:

    ansatz_dicts_arr = results_dict["full_population"]
    eval_dicts = results_dict["full_eval_metrics"]
    full_pop_fitness.extend(results_dict["full_fitness"])
    avg_pop_fitness.append(np.average(results_dict["full_fitness"]))
    std_dev_pop_fitness.append(np.std(results_dict["full_fitness"]))

    for ansatz_dicts in ansatz_dicts_arr:
        correct_ansatz_dicts = []
        for moment_dict in ansatz_dicts:
            correct_ansatz_dicts.append({})
            for k, v in moment_dict.items():
                correct_ansatz_dicts[-1][int(k)] = v  

        full_pop_ansatz.append(gav.Individual(n_qubits, n_moments, genepool, rng_seed, ansatz_dicts=correct_ansatz_dicts))
    
    extra_arr = []
    for eval_dict in eval_dicts:
        full_pop_auroc.append(eval_dict["auroc"])
        extra_arr.append(eval_dict["auroc"])
    avg_pop_auroc.append(np.average(extra_arr))



# distances_from_best = gav.euclidean_distances(best_ansatz, half_pop_ansatz)
filepath_avg_fitness = os.path.join(
    '/Users/tsievert/Downloads/result_plots',
    "FULL_POP_avg_fitness.png"
)
plt.figure(0)
plt.style.use("seaborn")
plt.plot([i for i in range(len(avg_pop_fitness))], avg_pop_fitness, c='k', zorder=1)
plt.errorbar([i for i in range(len(avg_pop_fitness))], avg_pop_fitness, yerr=std_dev_pop_fitness, zorder=2)
plt.scatter([i for i in range(len(avg_pop_fitness))], avg_pop_fitness, marker="o", c=avg_pop_auroc, cmap=plt.set_cmap('plasma'), zorder=3)
cbar = plt.colorbar()
cbar.set_label("Avg. AUROC")
plt.ylabel("Avg. Fitness")
plt.xlabel("Generation")
plt.title("Average Fitness Across Generations")
plt.savefig(filepath_avg_fitness, format="png")
plt.close(0)

filepath_full_fitness = os.path.join(
    '/Users/tsievert/Downloads/result_plots',
    "FULL_POP_full_fitness.png"
)
x_arr = []
for i in range(len(avg_pop_fitness)):
    for _ in range(30):
        x_arr.append(i)
plt.figure(1)
plt.style.use("seaborn")
plt.plot([i for i in range(len(avg_pop_fitness))], avg_pop_fitness, c='k', zorder=1)
plt.scatter(x_arr, full_pop_fitness, marker=".", c=full_pop_auroc, cmap=plt.set_cmap('plasma'), zorder=2)
cbar = plt.colorbar()
cbar.set_label("AUROC")
plt.ylabel("Fitness")
plt.xlabel("Generation")
plt.title("Fitness Across Generations")
plt.savefig(filepath_full_fitness, format="png")
plt.close(1)

perplexities = [i for i in range(2, 84)]
for perplexity in perplexities:

    data_tsne = gav.tsne(full_pop_ansatz, rng_seed=2, perplexity=perplexity)
    x, y = data_tsne[0], data_tsne[1]
    filepath_tsne = os.path.join(
        '/Users/tsievert/Downloads/result_plots',
        "FULL_POP_tsne_distance_data_perp%2d.png"
        % perplexity
    )
    plt.figure(2)
    plt.style.use("seaborn")
    plt.scatter(x, y, marker=".", c=full_pop_auroc, cmap=plt.set_cmap('plasma'))
    plt.ylabel("a.u.")
    plt.xlabel("a.u.")
    cbar = plt.colorbar()
    cbar.set_label("AUROC")
    plt.title("tSNE of Population")
    plt.savefig(filepath_tsne, format="png")
    plt.close(2)