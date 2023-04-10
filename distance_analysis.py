import os

import ga_vqc as gav
import matplotlib.pyplot as plt
import numpy as np
import json

dict_arr = []
for filename in os.listdir(''):
   with open(os.path.join(os.getcwd(), filename), 'r') as f: # open in readonly mode
        dict_arr.append(json.load(f))

best_ansatz = gav.Individual(dict_arr[-1]["best_ansatz"])

full_pop_ansatz = []
full_pop_auroc = []
for results_dict in dict_arr:
    ansatz_dicts = results_dict["full_population"]
    eval_dicts = results_dict["full_eval_metrics"]
    for ansatz_dict in ansatz_dicts:   
        full_pop_ansatz.append(gav.Individual(ansatz_dict=ansatz_dict))
    for eval_dict in eval_dicts:
        full_pop_auroc.append(eval_dict["auroc"])

perplexities = [2, 5, 10, 15, 30]
for perplexity in perplexities:
    data_tsne = gav.tsne(full_pop_ansatz, rng_seed=2, perplexity=perplexity)
    x, y = data_tsne[0], data_tsne[1]
    filepath_tsne = os.path.join(
        '',
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

distances_from_best = gav.euclidean_distances(best_ansatz, full_pop_ansatz)
filepath_euclid = os.path.join(
    '',
    "FULL_POP_euclid_distance_data.png"
)
plt.figure(0)
plt.style.use("seaborn")
plt.scatter(distances_from_best, full_pop_auroc, marker=".", color="g")
plt.ylabel("AUROC")
plt.xlabel("Euclidian distance from best ansatz")
plt.title("Euclidean Distances from Best Performing Ansatz")
plt.savefig(filepath_euclid, format="png")
plt.close(0)