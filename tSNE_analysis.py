import os

import ga_vqc as gav
import matplotlib.pyplot as plt
import numpy as np
import json

dict_arr = []
for filename in os.listdir('/Users/tsievert/GA Caltech/QAE_4_HEP/ga_runs/run-2023-04-09_01-12-46'):
   with open(os.path.join(os.getcwd(), filename), 'r') as f: # open in readonly mode
        dict_arr.append(json.load(f))

full_pop_ansatz = []
full_pop_auroc = []
for results_dict in dict_arr:
    ansatz_dicts = results_dict["full_population"]
    eval_dicts = results_dict["full_eval_metrics"]
    for ansatz_dict in ansatz_dicts:   
        full_pop_ansatz.append(gav.Individual(ansatz=ansatz_dict))
    for eval_dict in eval_dicts:
        full_pop_auroc.append(eval_dict["auroc"])

data_tsne = gav.tsne(full_pop_ansatz, rng_seed=2, perplexity=30)
x, y = data_tsne[0], data_tsne[1]
filepath_tsne = os.path.join(
    '/Users/tsievert/GA Caltech/QAE_4_HEP/ga_curves/run-2023-04-09_01-12-46',
    "FULL_POP_tsne_distance_data.png"
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