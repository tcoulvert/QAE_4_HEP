import datetime

import ga_vqc as gav
import pennylane.numpy as np
from sklearn.preprocessing import MinMaxScaler

from qae import main

### Making the training data ###
events = np.load("./Data/10k_dijet.npy", requires_grad=False)
scaler = MinMaxScaler(feature_range=(0, np.pi))
events = scaler.fit_transform(events)

### Making the validation data ###
events_bb1 = np.load("./Data/10k_dijet_bb1.npy", requires_grad=False)
events_bb1 = scaler.fit_transform(events_bb1)

classes = np.load("./Data/10k_dijet_bb1_class.npy", requires_grad=False)
f = open("./Data/events_LHCO2020_BlackBox1.masterkey", "r")
event_classes = np.genfromtxt(f, delimiter=",")
event_class = event_classes[classes.tolist()]

sig_event_ixs = np.nonzero(event_class, requires_grad=False)[0]
bkg_event_ixs = np.where(event_class == 0)[0]

rng_seed = 42
rng = np.random.default_rng(rng_seed)
chosen_bkg_event_ixs = rng.choice(
    bkg_event_ixs, 1000 - np.size(sig_event_ixs), replace=False
)

chosen_ixs = np.concatenate((sig_event_ixs, chosen_bkg_event_ixs))

chosen_val_events = events_bb1[chosen_ixs, :]
chosen_val_class = event_class[chosen_ixs]

### Make the Gate array ###
gates_dict = {"I": (1, 0), "RY": (1, 1), "CNOT": (2, 0)}
gates_probs = [0.15, 0.6, 0.25]
genepool = gav.Genepool(gates_dict, gates_probs)

ansatz_dicts = [{0: "RY", 1: "RY", 2: "RY"}, 
    {0: "CNOT_C-1", 1: "CNOT_T-0", 2: "I"}, 
    {0: "CNOT_C-2", 1: "I", 2: "CNOT_T-0"}, 
    {0: "I", 1: "CNOT_C-2", 2: "CNOT_T-1"}
]
n_ansatz_qubits = 3
ansatz = gav.Individual(n_ansatz_qubits, len(ansatz_dicts), genepool, rng_seed, ansatz_dicts=ansatz_dicts)
vqc_config_ansatz = {
    "n_wires": 6,  # allows us to use GA to optimize subsets of a circuit
    "n_trash_qubits": 2,
    "n_latent_qubits": 1,
    "n_shots": 1000,  # ~1000
    "events": events,
    "batch_size": 32,  # powers of 2, between 1 to 32
    "GPU": False,
    "events_val": chosen_val_events,
    "truth_val": chosen_val_class,
    "rng_seed": rng_seed,
}
ansatz.convert_to_qml()
ansatz.draw_ansatz()
vqc_config_ansatz["ansatz_dicts"] = ansatz.ansatz_dicts
vqc_config_ansatz["ansatz_qml"] = ansatz.ansatz_qml
vqc_config_ansatz["params"] = ansatz.params
vqc_config_ansatz["n_ansatz_qubits"] = n_ansatz_qubits
vqc_config_ansatz["start_time"] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
vqc_config_ansatz["gen"] = 0

fitness_arr = []
auroc_arr = []
for i in range(20):
    vqc_config_ansatz["ix"] = i
    output_dict = main(vqc_config_ansatz)
    fitness_arr.append(output_dict["fitness_metric"])
    auroc_arr.append(output_dict["eval_metrics"]["auroc"])
print(f"Final fitness distribution: {fitness_arr}")
print(f"Avg fitness: {np.mean(fitness_arr)},  Std Dev: {np.std(fitness_arr)}, Std Dev of Mean: {np.std(fitness_arr) / (20**0.5)}")
print(f"Final AUROC distribution: {auroc_arr}")
print(f"Avg AUROC: {np.mean(auroc_arr)},  Std Dev: {np.std(auroc_arr)}, Std Dev of Mean: {np.std(auroc_arr) / (20**0.5)}")