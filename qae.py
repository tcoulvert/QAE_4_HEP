import contextlib
import copy
import os
import time
import math

import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
from pickle import dump

from sklearn.metrics import roc_auc_score, roc_curve


def main(config) -> {"fitness_metric": int, "eval_metrics": {}}:
    if config["GPU"]:
        time.sleep(config['ix'])
        with contextlib.redirect_stdout(None):
            exec('import setGPU')

        dev = qml.device(
            "qulacs.simulator",
            wires=config["n_wires"],
            gpu=config["GPU"],
            shots=config["n_shots"],
        )
    else:
        dev = qml.device(
            "default.qubit", wires=config["n_wires"], shots=config["n_shots"]
        )

    config["qnode"] = qml.QNode(circuit, dev, diff_method="best")

    config["swap_pattern"] = compute_swap_pattern(
        config["n_ansatz_qubits"],
        config["n_latent_qubits"],
        config["n_trash_qubits"],
        config["n_wires"],
    )

    print(f"running circuit {config['ix']}")
    print(config["diagram"] + '\n\n')

    vqc_output = train(config)
    return vqc_output


def compute_swap_pattern(n_ansatz_qubits, n_latent_qubits, n_trash_qubits, n_wires):
    swap_pattern = []
    for i in range(n_trash_qubits):
        single_swap = [n_wires - 1, 0, 0]
        single_swap[1] = n_latent_qubits + i
        single_swap[2] = n_ansatz_qubits + i

        swap_pattern.append(single_swap)

    return swap_pattern


def batch_events(events, batch_size, rng):
    rebatch_step = np.size(events, axis=0) // batch_size
    n_extra_choices = np.size(events, axis=0) % batch_size

    batched_events = np.zeros(
        (rebatch_step + n_extra_choices, batch_size, np.size(events, axis=1))
    )
    for i in range(rebatch_step):
        batched_events[i] = events[i * batch_size : (i + 1) * batch_size, :]
    if n_extra_choices != 0:
        batched_events[-1] = np.vstack(
            (
                events[n_extra_choices - batch_size :, :],
                rng.choice(events[: n_extra_choices - batch_size, :], n_extra_choices),
            )
        )

    return batched_events, rebatch_step


def circuit(params, event=None, config=None):
    # Embed the data into the circuit
    qml.broadcast(
        qml.RX,
        wires=range(config["n_latent_qubits"] + config["n_trash_qubits"]),
        pattern="single",
        parameters=event,
    )
    qml.Hadamard(wires=config["n_wires"] - 1)

    # Run the actual circuit ansatz
    for m in config["qml"]:
        exec(m)

    # Perform the SWAP-Test for a qubit fidelity measurement
    qml.broadcast(
        qml.CSWAP,
        wires=range(config["n_latent_qubits"], config["n_wires"]),
        pattern=config["swap_pattern"],
    )
    qml.Hadamard(wires=config["n_wires"] - 1)

    # qml.operation.Tensor(qml.PauliZ...) for measuring multiple anscillary wires
    return qml.expval(qml.PauliZ(wires=config["n_wires"] - 1))

def var_circuit(params, event=None, config=None):
    # Embed the data into the circuit
    qml.broadcast(
        qml.RX,
        wires=range(config["n_latent_qubits"] + config["n_trash_qubits"]),
        pattern="single",
        parameters=event,
    )
    qml.Hadamard(wires=config["n_wires"] - 1)

    # Run the actual circuit ansatz
    for m in config["qml"]:
        exec(m)

    # Perform the SWAP-Test for a qubit fidelity measurement
    qml.broadcast(
        qml.CSWAP,
        wires=range(config["n_latent_qubits"], config["n_wires"]),
        pattern=config["swap_pattern"],
    )
    qml.Hadamard(wires=config["n_wires"] - 1)

    # qml.operation.Tensor(qml.PauliZ...) for measuring multiple anscillary wires
    return qml.var(qml.PauliZ(wires=config["n_wires"] - 1))

def square_loss(fidelity):
    loss = (1 - fidelity)**2
    
    return loss

def cost(params, event, config):
    fidelity = config["qnode"](params, event=event, config=config)
    
    return square_loss(fidelity)

def train(config):
    def find_best_index(cost_arr):
        index = 0
        for i in range(len(cost_arr)):
            if np.std(cost_arr[i:i+5]) < 0.05:
                index = i
                break
        return index
    
    rng = np.random.default_rng(seed=config["rng_seed"])
    adm_cost = []
    adm_stddev = []
    adm_auroc = []
    opt = qml.AdamOptimizer(0.01, beta1=0.9, beta2=0.999)

    # Initialized to 2 b/c worst performance would be value of 1.
    best_perf = {
        "avg_loss": 2.0,
        "stddev_loss": 0.0,
        "opt_params": None,
        "auroc": 0.0,
    }
    step_size_factor = -1
    thetas = config["params"]
    thetas_arr = []
    step = 0
    wait_till_step = 0
    rebatch_step = 0

    while True:
        if step == rebatch_step:
            batched_events, rebatch_step = batch_events(
                rng.permutation(config["events"]), config["batch_size"], rng
            )

        events_batch = batched_events[step - rebatch_step]
        grads = []
        costs = []
        stddevs = []

        # iterating over all the training data
        for i in range(events_batch.shape[0]):
            (grad_i, _), cost_i = opt.compute_grad(cost, (thetas, events_batch[i], config), {})
            var_i = var_circuit(thetas, events_batch[i], config)

            grads.append(grad_i)
            costs.append(cost_i.item())
            stddevs.append(var_i.item()**(1/2))

        adm_auroc.append(compute_auroc(thetas, config))
        thetas_arr.append(copy.deepcopy(thetas))
        thetas = thetas - (10**step_size_factor * np.sum(grads, axis=0))
        adm_cost.append(np.mean(costs))
        adm_stddev.append(np.mean(stddevs))
        
        # if step%10 == 0:
        #     print(step)
            # print(adm_cost)
        step += 1

        # require some minimum fraction of epochs
        min_steps = np.min([20, len(config["events"])])
        if (step / len(config["events"])) < config["n_epochs"]:
            continue
        elif step < min_steps:
            continue
        elif step < wait_till_step:
            continue

        # ADD IN DECREASE STEP SIZE BY INCREMENTING DOWN THE STEP_SIZE_FACTOR
        if math.isclose(np.mean(adm_cost[-min_steps:min_steps/2]), 
            np.mean(adm_cost[-min_steps/2:]),
            abs_tol=np.max(adm_stddev[-min_steps:])
        ):
            if step_size_factor == -6:
                best_index = find_best_index(adm_cost)

                best_perf["opt_params"] = thetas_arr[best_index]
                best_perf["avg_loss"] = adm_cost[best_index]
                best_perf["stddev_loss"] = adm_stddev[best_index]
                best_perf["auroc"] = adm_auroc[best_index]

                break
            step_size_factor -= 1
            wait_till_step = step + 20

        
        if np.std(adm_cost[-min_steps:]) < 0.1:
            best_index = find_best_index(adm_cost)

            best_perf["opt_params"] = thetas_arr[best_index]
            best_perf["avg_loss"] = adm_cost[best_index]
            best_perf["stddev_loss"] = adm_stddev[best_index]
            best_perf["auroc"] = adm_auroc[best_index]

            break




    # Saving outputs
    script_path = os.path.dirname(os.path.realpath(__file__))
    destdir = os.path.join(script_path, "qae_runs", "run-%s" % config["start_time"])
    if not os.path.exists(destdir):
        os.makedirs(destdir)

    # Saving the params, costs, and aurocs computed for each of the 20 trials
    destdir_thetas = os.path.join(destdir, "opt_thetas")
    if not os.path.exists(destdir_thetas):
        os.makedirs(destdir_thetas)
    filepath_thetas = os.path.join(
        destdir_thetas,
        "%02d_%03d_%02d_ga_best%.e_data_theta"
        % (config["ix"], config["gen"], i, config["batch_size"]),
    )
    np.save(filepath_thetas, best_perf["opt_params"])
    
    destdir_metas = os.path.join(destdir, "metas")
    if not os.path.exists(destdir_metas):
        os.makedirs(destdir_metas)
    filepath_metas = os.path.join(
        destdir_metas,
        "%02d_%03d_%02d_ga_best%.e_data_metas"
        % (config["ix"], config["gen"], i, config["batch_size"]),
    )
    np.save(filepath_metas, [best_perf["avg_loss"], best_perf["stddev_loss"], best_perf["auroc"]])
    
    # destdir_aurocs = os.path.join(destdir, "aurocs")
    # if not os.path.exists(destdir_aurocs):
    #     os.makedirs(destdir_aurocs)
    # filepath_aurocs_arr = [os.path.join(
    #     destdir_aurocs,
    #     "%02d_%03d_%02d_ga_best%.e_data_aurocs"
    #     % (config["ix"], config["gen"], i, config["batch_size"]),
    # ) for i in range(config["n_retrains"])]
    # np.save(filepath_auroc, best_perf["auroc"])
    

    destdir_ansatz = os.path.join(destdir, "opt_ansatz")
    if not os.path.exists(destdir_ansatz):
        os.makedirs(destdir_ansatz)
    # Make ansatz represented in list-of-dicts-of-qubits
    filepath_ansatz = os.path.join(
        destdir_ansatz,
        "%02d_%03dga_best%.e_data_ansatz"
        % (config["ix"], config["gen"], config["batch_size"]),
    )
    with open(filepath_ansatz, "wb") as f:
        dump(config["dicts"], f)
    # Make ansatz to run using loop
    filepath_run = os.path.join(
        destdir_ansatz,
        "%02d_%03dga_best%.e_run_ansatz"
        % (config["ix"], config["gen"], config["batch_size"]),
    )
    np.save(filepath_run, config["qml"])
    # Make ansatz to draw in output files
    filepath_draw = os.path.join(
        destdir_ansatz,
        "%02d_%03dga_best%.e_draw_ansatz"
        % (config["ix"], config["gen"], config["batch_size"]),
    )
    with open(filepath_draw, "w") as f:
        f.write(config["diagram"])

    destdir_curves = os.path.join(destdir, "qml_curves")
    if not os.path.exists(destdir_curves):
        os.makedirs(destdir_curves)
    filepath_opt_loss = os.path.join(
        destdir_curves,
        "%02d_%03dga_ADAM_Descent-%d_data.png"
        % (config["ix"], config["gen"], config["batch_size"]),
    )
    plt.figure(0)
    plt.style.use("seaborn")
    plt.errorbar(adm_cost, yerr=adm_stddev, ls="None")
    plt.plot(adm_cost, "g", label="ADAM Descent - %d data" % config["batch_size"])
    plt.ylabel("Loss (1 - Fid.)")
    plt.xlabel("Optimization steps")
    plt.legend()
    plt.savefig(filepath_opt_loss, format="png")
    plt.close(0)

    filepath_opt_auroc = os.path.join(
        destdir_curves,
        "%02d_%03dga_auroc_bb1-%d_data.png"
        % (config["ix"], config["gen"], config["batch_size"]),
    )
    plt.figure(1)
    plt.style.use("seaborn")
    plt.plot(adm_auroc, "b", label="AUROC - %d data" % config["batch_size"])
    plt.ylabel("AUROC")
    plt.xlabel("Optimization steps")
    plt.legend()
    plt.savefig(filepath_opt_auroc, format="png")
    plt.close(1)

    auroc, bkg_rejec, tpr = compute_auroc(best_perf["opt_params"], config, FINAL=True)
    filepath_auroc = os.path.join(
        destdir_curves,
        "%02d_%03dga_roc_bb1-%d_data.png"
        % (config["ix"], config["gen"], config["batch_size"]),
    )
    plt.figure(2)
    plt.style.use("seaborn")
    plt.plot(bkg_rejec, tpr, label=f"AUROC - {auroc}")
    plt.title("ROC on BB1 w/ GA Ansatz")
    plt.xlabel("Bkg. Rejection")
    plt.ylabel("Sig.  Acceptance")
    plt.legend()
    plt.savefig(filepath_auroc, format="png")
    plt.close(2)

    filepath_combine = os.path.join(
        destdir_curves,
        "%02d_%03dga_combine-%d_data.png"
        % (config["ix"], config["gen"], config["batch_size"]),
    )
    fig, ax1 = plt.subplots()
    plt.style.use("seaborn-v0_8-dark")
    ax2 = ax1.twinx()
    ax1.errorbar(adm_cost, yerr=adm_stddev, ls="None")
    ax1.plot(adm_cost, "g", label="ADAM Descent - %d data" % config["batch_size"])
    ax2.plot(adm_auroc, "b", label="AUROC - %d data" % config["batch_size"])
    ax1.set_xlabel('Optimization steps', color = 'k')
    ax1.set_ylabel('Loss (1 - Fid.)', color = 'g')
    ax2.set_ylabel('AUROC', color = 'b')
    plt.savefig(filepath_combine, format="png")
    plt.close()

    # return {
    #     "fitness_metric": 1 - best_perf["avg_loss"],
    #     "eval_metrics": {
    #         "auroc": best_perf["auroc"],
    #     },
    # }
    return {
        "fitness_metrics": {
            "avg_fitness": best_perf["avg_loss"],
            "stddev_fitness": best_perf["stddev_loss"],
        },
        "eval_metrics": {
            "avg_auroc": best_perf["auroc"],
        },
    }


def compute_auroc(thetas, config, FINAL=False):
    costs = []
    for i in range(np.size(config["events_val"], axis=0)):
        costs.append(
            cost(thetas, config["events_val"][i], config).item()
        )

    # Don't do "1 - np.array(...)" b/c 0 is bkg and 1 is sig
    fid_pred = np.array(costs, requires_grad=False)
    auroc = roc_auc_score(config["truth_val"], fid_pred).item()

    if FINAL:
        fpr, tpr, thresholds = roc_curve(config["truth_val"], fid_pred)
        # bkg_rejec = 1 - fpr
        bkg_rejec = fpr

        fid_split = [None, None]
        bkg_cost, sig_cost = [], []

        for i in range(np.size(config["truth_val"], axis=0)):
            if config["truth_val"][i] == 0:
                bkg_cost.append(fid_pred[i])
            elif config["truth_val"][i] == 1:
                sig_cost.append(fid_pred[i])

        color_arr = ['r', 'b']
        for i in range(2):
            if i == 0:
                label_str = 'bkg'
                fid_split[i] = np.array(bkg_cost)
                n_bins = 100
            elif i == 1:
                label_str = 'sig'
                fid_split[i] = np.array(sig_cost)
                n_bins = 10
            
            plt.figure(4)
            plt.hist(fid_split[i], bins=n_bins, density=True, 
                color=color_arr[i], alpha=0.5, linewidth=1.7,
                label=label_str
            )
            plt.legend(loc='upper left')
            plt.title('Fidelities on BB1')
            plt.style.use("seaborn")
            plt.xlabel('Fid.')
            plt.ylabel('a.u')
            
        ### Saving outputs ###
        script_path = os.path.dirname(os.path.realpath(__file__))
        destdir = os.path.join(script_path, "qae_runs", "run-%s" % config["start_time"])
        if not os.path.exists(destdir):
            os.makedirs(destdir)
        destdir_curves = os.path.join(destdir, "qml_curves")
        if not os.path.exists(destdir_curves):
            os.makedirs(destdir_curves)
        filepath_hist = os.path.join(
            destdir_curves,
            "%02d_%03dsplit_fid_hist_bb1-%d.png"
            % (config["ix"], config["gen"], config["batch_size"])
        )
        plt.savefig(filepath_hist, format='png')
        plt.close(4)

        return auroc, bkg_rejec, tpr

    return auroc
