import os

import ga_vqc as gav
import matplotlib.pyplot as plt
import numpy as np
import json

n_qubits = 3
n_moments = 4
gates_dict = {"I": (1, 0), "U3": (1, 3), "CNOT": (2, 0)}
gates_probs = [0.15, 0.6, 0.25]
genepool = gav.Genepool(gates_dict, gates_probs)
rng_seed = 4

dir = '/Users/tsievert/Downloads/evo_results_upto_15'


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
half_pop_ansatz = []
full_pop_auroc = []
half_pop_auroc = []
half_pop_fitness = []

for results_dict in dict_arr:

    ansatz_dicts_arr = results_dict["full_population"]
    eval_dicts = results_dict["full_eval_metrics"]
    half_pop_fitness.extend(results_dict["full_fitness"])

    count = 0
    for ansatz_dicts in ansatz_dicts_arr:
        correct_ansatz_dicts = []
        for moment_dict in ansatz_dicts:
            correct_ansatz_dicts.append({})
            for k, v in moment_dict.items():
                correct_ansatz_dicts[-1][int(k)] = v  

        full_pop_ansatz.append(gav.Individual(n_qubits, n_moments, genepool, rng_seed, ansatz_dicts=correct_ansatz_dicts))
        if count > 15:
            half_pop_ansatz.append(full_pop_ansatz[-1])
        count += 1
    
    count = 0
    for eval_dict in eval_dicts:
        full_pop_auroc.append(eval_dict["auroc"])
        if count > 15:
            half_pop_auroc.append(full_pop_auroc[-1])
        count += 1


distances_from_best = gav.euclidean_distances(best_ansatz, half_pop_ansatz)
filepath_euclid = os.path.join(
    '/Users/tsievert/Downloads/evo_results_upto_15',
    "FULL_POP_euclid_distance_data.png"
)
plt.figure(0)
plt.style.use("seaborn")
plt.scatter(distances_from_best, half_pop_fitness, marker=".", c=half_pop_auroc, cmap=plt.set_cmap('plasma'))
cbar = plt.colorbar()
cbar.set_label("AUROC")
plt.ylabel("Fitness")
plt.xlabel("Euclidian distance from best ansatz")
plt.title("Euclidean Distances from Best Performing Ansatz")
plt.savefig(filepath_euclid, format="png")
plt.close(0)

perplexities = [i*2 for i in range(1, 20)]
for perplexity in perplexities:

    data_tsne = gav.tsne(full_pop_ansatz, rng_seed=2, perplexity=perplexity)
    x, y = data_tsne[0], data_tsne[1]
    filepath_tsne = os.path.join(
        '/Users/tsievert/Downloads/result_plots',
        "FULL_POP_tsne_distance_data_perp%2d.png"
        % perplexity
    )
    plt.figure(1)
    plt.style.use("seaborn")
    plt.scatter(x, y, marker=".", c=full_pop_auroc, cmap=plt.set_cmap('plasma'))
    plt.ylabel("a.u.")
    plt.xlabel("a.u.")
    cbar = plt.colorbar()
    cbar.set_label("AUROC")
    plt.title("tSNE of Current Population")
    plt.savefig(filepath_tsne, format="png")
    plt.close(1)