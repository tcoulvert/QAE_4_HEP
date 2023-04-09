# import contextlib
import os
import matplotlib.pyplot as plt

import pennylane as qml

from pennylane import numpy as np
from pickle import dump

from sklearn.metrics import roc_auc_score, roc_curve


def main(config):
    if config["GPU"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{(config['ix']+2)%8}"
        # time.sleep(ix)
        # with contextlib.redirect_stdout(None):
        #     exec('import setGPU')

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
    print(config["ansatz_dicts"])

    best_fid = train(config)
    return best_fid


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
    for m in config["ansatz_qml"]:
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

def square_loss(fidelities):
    loss = (1 - fidelities)**2
    
    return loss

def cost(params, event, config):
    fidelities = config["qnode"](params, event=event, config=config)
    
    return square_loss(fidelities)

def train(config):
    rng = np.random.default_rng(seed=config["rng_seed"])
    adm_cost = []
    adm_auroc = []
    opt = qml.AdamOptimizer(0.01, beta1=0.9, beta2=0.999)

    # Initialized to 2 b/c worst performance would be value of 1.
    best_perf = {
        "avg_loss": 2.0,
        "opt_params": None,
        "auroc": 0.0,
    }
    stop_check = {
        "old_avg": 0.0,
        "old_std_dev": 0.0,
        "new_avg": 0.0,
        "new_std_dev": 0.0,
    }
    stop_check_factor = 40
    step_size_factor = -1
    thetas = config["params"]
    step = 0
    rebatch_step = 0

    while True:
        if step == rebatch_step:
            batched_events, rebatch_step = batch_events(
                rng.permutation(config["events"]), config["batch_size"], rng
            )

        events_batch = batched_events[step - rebatch_step]
        grads = []
        costs = []

        # iterating over all the training data
        for i in range(events_batch.shape[0]):
            (grad_i, _), cost_i = opt.compute_grad(cost, (thetas, events_batch[i], config), {})
            grads.append(grad_i)
            costs.append(cost_i.item())

        if best_perf["avg_loss"] > np.mean(costs, axis=0):
            best_perf["avg_loss"] = np.mean(costs, axis=0).item()
            best_perf["opt_params"] = thetas
            auroc = compute_auroc(thetas, config)
            best_perf["auroc"] = auroc
            adm_auroc.append(auroc)
        else:
            adm_auroc.append(adm_auroc[-1])
        thetas = thetas - (10**step_size_factor * np.sum(grads, axis=0))
        adm_cost.append(np.mean(costs, axis=0))

        # checking the stopping condition
        if step > stop_check_factor:
            stop_check["old_avg"] = np.mean(adm_cost[-40:-20], axis=0)
            stop_check["old_std_dev"] = np.std(adm_cost[-40:-20], axis=0)
            stop_check["new_avg"] = np.mean(adm_cost[-20:], axis=0)
            stop_check["new_std_dev"] = np.std(adm_cost[-20:], axis=0)
            if np.isclose(
                stop_check["old_avg"],
                stop_check["new_avg"],
                atol=np.amax([stop_check["old_std_dev"], stop_check["new_std_dev"]]),
            ):
                step_size_factor -= 1
                stop_check_factor = step + 20
                if step_size_factor < -8:
                    break

        step += 1

    # Saving outputs
    script_path = os.path.dirname(os.path.realpath(__file__))
    destdir = os.path.join(script_path, "qae_runs", "run-%s" % config["start_time"])
    if not os.path.exists(destdir):
        os.makedirs(destdir)

    destdir_thetas = os.path.join(destdir, "opt_thetas")
    if not os.path.exists(destdir_thetas):
        os.makedirs(destdir_thetas)
    filepath_thetas = os.path.join(
        destdir_thetas,
        "%02d_%03dga_best%.e_data_theta"
        % (config["ix"], config["gen"], config["batch_size"]),
    )
    np.save(filepath_thetas, best_perf["opt_params"])

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
        dump(config["ansatz_dicts"], f)
    # Make ansatz to run using loop
    filepath_run = os.path.join(
        destdir_ansatz,
        "%02d_%03dga_best%.e_run_ansatz"
        % (config["ix"], config["gen"], config["batch_size"]),
    )
    np.save(filepath_run, config["ansatz_qml"])
    # Make ansatz to draw in output files
    filepath_draw = os.path.join(
        destdir_ansatz,
        "%02d_%03dga_best%.e_draw_ansatz"
        % (config["ix"], config["gen"], config["batch_size"]),
    )
    ansatz_draw = qml.draw(config["qnode"], decimals=None, expansion_strategy="device")(
        thetas, event=events_batch[0], config=config
    )
    with open(filepath_draw, "w") as f:
        f.write(ansatz_draw)

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
    plt.plot(adm_auroc, "g", label="AUROC - %d data" % config["batch_size"])
    plt.ylabel("AUROC")
    plt.xlabel("Optimization steps")
    plt.legend()
    plt.savefig(filepath_opt_auroc, format="png")
    plt.close(1)

    auroc, bkg_rejec, tpr = compute_auroc(thetas, config, FINAL=True)
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

    return {
        "fitness_metric": 1 - best_perf["avg_loss"],
        "eval_metrics": {
            "auroc": best_perf["auroc"],
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
        bkg_rejec = 1 - fpr
        return auroc, bkg_rejec, tpr

    return auroc
