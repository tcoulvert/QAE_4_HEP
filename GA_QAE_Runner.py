#! usr/bin/env python3

import argparse
import os

import scipy as sp
from pennylane import numpy as np

# import numpy
from sklearn.preprocessing import MinMaxScaler

import ga_vqc as gav
from qae import main as qae_main


def main(rng_seed):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    ### Making the training data ###
    events = np.load(os.path.join(dir_path, "Data/10k_dijet.npy"), requires_grad=False)
    scaler = MinMaxScaler(feature_range=(0, sp.pi))
    events = scaler.fit_transform(events)

    ### Making the validation data ###
    events_bb1 = np.load(os.path.join(dir_path, "Data/10k_dijet_bb1.npy"), requires_grad=False)
    events_bb1 = scaler.fit_transform(events_bb1)

    classes = np.load(os.path.join(dir_path, "Data/10k_dijet_bb1_class.npy"), requires_grad=False)
    f = open(os.path.join(dir_path, "Data/events_LHCO2020_BlackBox1.masterkey"), "r")
    event_classes = np.genfromtxt(f, delimiter=",")
    event_class = event_classes[classes.tolist()]

    sig_event_ixs = np.nonzero(event_class, requires_grad=False)[0]
    bkg_event_ixs = np.where(event_class == 0)[0]

    rng = np.random.default_rng(rng_seed)
    chosen_bkg_event_ixs = rng.choice(
        bkg_event_ixs, 1000 - np.size(sig_event_ixs), replace=False
    )

    chosen_ixs = np.concatenate((sig_event_ixs, chosen_bkg_event_ixs))

    chosen_val_events = events_bb1[chosen_ixs, :]
    chosen_val_class = event_class[chosen_ixs]

    ### Make the Gate array ###
    gates_dict = {"I": (1, 0), "RX": (1, 1), "RY": (1, 1), "RZ": (1, 1), "PhaseShift": (1, 1), "CNOT": (2, 0)}
    gates_probs = [0.15, 0.15, 0.15, 0.15, 0.15, 0.25]
    genepool = gav.Genepool(gates_dict, gates_probs)
    
    vqc_config = {
        "n_wires": 6,  # allows us to use GA to optimize subsets of a circuit
        "n_trash_qubits": 2,
        "n_latent_qubits": 1,
        "n_shots": 100,  # ~1000
        "events": events,
        "batch_size": 8,  # powers of 2, between 1 to 32
        "GPU": False,
        "events_val": chosen_val_events,
        "truth_val": chosen_val_class,
        "rng_seed": rng_seed,
    }

    ga_output_path = os.path.dirname(os.path.realpath(__file__))

    config = gav.Config(qae_main, vqc_config, genepool, ga_output_path)
    config.init_pop_size = 1000
    config.pop_size = 10
    config.max_moments = 4

    ga = gav.setup(config)
    ga.evolve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("-s", "--seed", type=int, help="Random numner generator seed")
    args = parser.parse_args()

    main(args.seed)
