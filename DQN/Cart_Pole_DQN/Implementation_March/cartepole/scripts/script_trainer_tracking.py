import os
import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from learning.trainer import Trainer

def plot_metric_compare(values, param_name, syn_index, metric_name, ap_indices=(0,1)):
    """Plot multiple AP indices of the same metric on a single axes with legend."""
    plt.figure()
    data = values[param_name][syn_index][metric_name]
    plt.plot(data, label=f"AP None")
    plt.title(f"{param_name} | Synapse {syn_index} | {metric_name}")
    plt.xlabel("Step")
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_metric(values):
    """Plot multiple AP indices of the same metric on a single axes with legend."""
    plt.figure()
    data = values
    plt.plot(data, label=f"Loss")
    plt.title(f"Loss (compare APs)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.show()

hyperparams = {
    "discount_factor": 0.90,
    "batch_size": 64,
    "max_episodes": 100,
    "max_steps": 200,
    "epsilon_max": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.000005,
    "memory_capacity": 1000000,
    "train": True
}
controller_hyperparams = {
    "a": 1.566e-8,
    "b": 0.350e-8,
    "c": 1e6,
    "g_threshold": 0.350e-8,
    "sigma_pulse_noise": 0.0,
    "scaling_factor": 5e7,
    "n_problem": 1,
}


def main():
    if hyperparams["train"]:
        trainer = Trainer(hyperparams, controller_hyperparams)
        trainer.train()
        tracked = trainer.agent.weight_controller.track_values
        loss = trainer.agent.loss_history
        with open("tracked.pkl", "wb") as f:
            pickle.dump(tracked, f)
        with open("loss.pkl", "wb") as f:
            pickle.dump(loss, f)
    else:
        with open("tracked.pkl", "rb") as f:
            tracked = pickle.load(f)
        with open("loss.pkl", "rb") as f:
            loss = pickle.load(f)
        # Plot both AP indices on the same plot for easy comparison
        plot_metric_compare(tracked, "FC.0.weight", (0,0), "weight")
        plot_metric(loss)
        # print(track_values["FC.0.weight"][(0, 0)]["weight"])
        # print(track_values["FC.0.weight"][(0, 0)]["g_ap"])
        # print(track_values["FC.0.weight"][(0, 0)]["g_bias"])
        # print(track_values["FC.0.weight"][(0, 0)]["x_index"])
        # print(track_values["FC.0.weight"][(0, 0)]["bias_index"])


if __name__ == "__main__":
    main()