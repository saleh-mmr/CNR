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



hyperparams = {
    "discount_factor": 0.99,
    "batch_size": 1000,
    "warmup_size": 1000,
    "network_size": 40,
    "max_steps_per_episode": 100,
    "max_episodes": 8000,
    "epsilon_max": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.00001,
    "memory_capacity": 10000,
    "g_ap": 18.0,
    "g_p": 15.0,
    "shift_parameter": 20,
    "g_bias": 30.0,
    "noise_stddev": 0.001,
    "CP_pole_length_1": 5.0,
    "CP_pole_mass_1": 1.0,
    "CP_pole_length_2": 10.0,
    "CP_pole_mass_2": 2.0,
    "CP_pole_length_3": 20.0,
    "CP_pole_mass_3": 5.0,
}



if __name__ == "__main__":
    seed = 837
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = Path(f"run_{timestamp}")
    # Create the folder
    folder.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    trainer = Trainer(hyperparams, seed, folder)
    rewards_mc1, rewards_mc2, rewards_mc3 = trainer.train()

    # Plot results
    fig, axes = plt.subplots(1, 1, figsize=(14, 6))

    axes[0].plot(rewards_mc1, label="Reward", linewidth=4)
    axes[0].set_xlabel("Episode", fontsize=15)
    axes[0].set_ylabel("Reward", fontsize=15)
    axes[0].set_title(f"Training on Cartpole Problems with Pole Length {hyperparams["CP_pole_length_1"]}, {hyperparams["CP_pole_length_2"]}, {hyperparams["CP_pole_length_3"]}", fontsize=16)
    axes[0].grid(True)

    plt.tight_layout()
    plot_path = folder / "training_plot.png"
    plt.savefig(plot_path, dpi=300)
    plt.show()
