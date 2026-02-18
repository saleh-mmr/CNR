import os
import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from learning.trainer import Trainer

def plot_metric(values, param_name, syn_index, metric_name):
    """
    values : loaded dictionary
    param_name   : e.g. "1.weight"
    syn_index    : integer index (e.g. 0)
    metric_name  : e.g. "weight", "loss", "g_ap"
    """
    data = values[param_name][syn_index][metric_name]
    plt.figure()
    plt.plot(data)
    plt.xlabel("Step")
    plt.ylabel(metric_name)
    plt.title(f"{param_name} | Synapse {syn_index} | {metric_name}")
    plt.grid(True)
    plt.show()


def plot_all_metrics(values, param_name, syn_index, metric_list):

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.flatten()

    for i, metric_name in enumerate(metric_list):
        data = values[param_name][syn_index][metric_name]

        ax = axes[i]
        ax.plot(data)
        ax.set_title(metric_name, fontsize=12)
        ax.set_xlabel("Step")
        ax.set_ylabel(metric_name)
        ax.grid(True)

        # Make subplot square
        ax.set_box_aspect(1)

    # Remove unused subplot (6th one)
    if len(metric_list) < 6:
        for j in range(len(metric_list), 6):
            fig.delaxes(axes[j])

    # Better spacing
    fig.suptitle(f"{param_name} | Synapse {syn_index}",
                 fontsize=16,
                 y=0.98)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    plt.show()

hyperparams = {
    "discount_factor": 0.99,
    "batch_size": 64,
    "max_episodes": 1000,
    "max_steps": 500,
    "epsilon_max": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.00005,
    "memory_capacity": 10000,
}
trainer = Trainer(hyperparams)
trainer.train()
# tracked = trainer.agent.weight_controller.track_values
# with open("track_values.pkl", "wb") as f:
#     pickle.dump(tracked, f)

# with open("track_values.pkl", "rb") as f:
#     track_values = pickle.load(f)
# plot_metric(track_values,"FC.0.weight", (0, 1), "weight")
# plot_metric(track_values,"FC.0.weight", (0, 1), "g_ap")
# plot_metric(track_values,"FC.0.weight", (0, 1), "g_bias")
# plot_metric(track_values,"FC.0.weight", (0, 1), "x_index")
# plot_metric(track_values,"FC.0.weight", (0, 1), "bias_index")
# print(track_values["FC.0.weight"][(0, 1)]["weight"][-10:])  # Print last 10 weight values for synapse (0, 1) in FC.0.weight
# metrics = ["weight", "g_ap", "g_bias", "x_index", "bias_index"]
# plot_all_metrics(track_values, "FC.0.weight", (15,2), metrics)
