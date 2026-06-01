import itertools
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from matplotlib import pyplot as plt
import random
import numpy as np
import torch
from learning.trainer import Trainer
from datetime import datetime
from pathlib import Path


network_size = [80, 256]
batch_size = [22]



base_hyperparams = {
    "discount_factor": 0.99,
    "batch_size": 128,
    "warmup_size": 128,
    "network_size": 128,
    "max_steps_per_episode": 100,
    "max_episodes": 200000,
    "epsilon_max": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.00001,
    "memory_capacity": 10000,

    "g_ap": 26.0,
    "g_p": 22.0,
    "shift_parameter": 20,
    "g_bias": 54.0,

    "noise_stddev": 0.0,
    "CP_pole_length_1": 2.8,
    "CP_pole_mass_1": 0.3,
    "CP_pole_length_2": 16.0,
    "CP_pole_mass_2": 0.6,
    "CP_pole_length_3": 24.0,
    "CP_pole_mass_3": 0.9,
}


if __name__ == "__main__":
    seed = 873
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    root_folder = Path("../weights/grid_search") / f"grid_{timestamp}"
    root_folder.mkdir(parents=True, exist_ok=True)

    for g_ap, g_p, g_bias, CP_pole_length_1  in itertools.product(G_ap, G_p, G_bias, CP_pole_length_1):

        hyperparams = base_hyperparams.copy()
        hyperparams["g_ap"] = float(g_ap)
        hyperparams["g_p"] = float(g_p)
        hyperparams["g_bias"] = float(g_bias)
        hyperparams["CP_pole_length_1"] = float(CP_pole_length_1)

        run_name = f"g_ap_{g_ap}_g_p_{g_p}_g_bias_{g_bias}_CP_pole_length_1_{CP_pole_length_1}"
        folder = root_folder / run_name
        folder.mkdir(parents=True, exist_ok=True)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        print(f"Running: {run_name}")

        trainer = Trainer(hyperparams, seed, folder)
        rewards = trainer.train()

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(rewards, label="Reward", linewidth=4)

        ax.set_xlabel("Episode", fontsize=15)
        ax.set_ylabel("Reward", fontsize=15)
        ax.set_title(
            f"g_ap={g_ap}, g_p={g_p}, g_bias={g_bias}",
            fontsize=16
        )
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        plot_path = folder / "training_plot.png"
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)