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
    ### Making the training data ###
    events = np.load("10k_dijet.npy", requires_grad=False)
    scaler = MinMaxScaler(feature_range=(0, sp.pi))
    events = scaler.fit_transform(events)
    ### ---------------------------- ###

    ### Making the validation data ###
    events_bb1 = np.load("10k_dijet_bb1.npy", requires_grad=False)
    events_bb1 = scaler.fit_transform(events_bb1)

    classes = np.load("10k_dijet_bb1_class.npy", requires_grad=False)
    f = open("events_LHCO2020_BlackBox1.masterkey", "r")
    event_classes = np.genfromtxt(f, delimiter=",")
    event_class = event_classes[classes.tolist()]

    sig_event_ixs = np.nonzero(event_class, requires_grad=False)[0]
    bkg_event_ixs = np.where(event_class == 0)[0]

    # rng = np.radnom.default_generator(rng_seed)
    rng = np.radnom.default_generator(42)
    chosen_bkg_event_ixs = rng.choice(
        bkg_event_ixs, 1000 - np.size(sig_event_ixs), replace=False
    )

    chosen_ixs = np.concatenate((sig_event_ixs, chosen_bkg_event_ixs))

    chosen_val_events = events_bb1[chosen_ixs, :]
    chosen_val_class = event_class[chosen_ixs]
    ### ---------------------------- ###

    config = {
        "backend_type": "high",
        "vqc": qae_main,  # main func that handles variational quantum circuit training
        "max_concurrent": 2,
        "n_qubits": 3,
        "max_moments": 4,
        "add_moment_prob": 0.15,
        "gates_arr": ["I", "RX", "RY", "RZ", "CNOT"],
        "gates_probs": [0.175, 0.175, 0.175, 0.175, 0.3],
        # "gates_arr": ["I", "RX", "RY", "RZ", "PhaseShift", "CNOT"],
        # "gates_probs": [0.15, 0.15, 0.15, 0.15, 0.15, 0.25],
        "pop_size": 20,  # must be a multiple of max_concurrent
        "init_pop_size": 1000,
        "n_new_individuals": 10,
        "n_winners": 10,  # needs to be an even number
        "n_mutations": 1,
        "n_mate_swaps": 1,
        "n_steps": 15,
        "rng_seed": rng_seed,
        "ga_output_path": os.path.dirname(os.path.realpath(__file__)),
        "vqc_config": {
            "n_wires": 6,  # allows us to use GA to optimize subsets of a circuit
            "n_trash_qubits": 2,
            "n_latent_qubits": 1,
            "n_shots": 500,
            "events": events,
            "batch_size": 32,
            "GPU": False,
            "events_val": chosen_val_events,
            "truth_val": chosen_val_class,
            "rng_seed": rng_seed,
        },
    }

    ga = gav.setup(config)
    ga.evolve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, help="Random numner generator seed")
    args = parser.parse_args()

    main(args.seed)
