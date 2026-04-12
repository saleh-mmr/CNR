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
    "warmup_size": 2000,
    "network_size": 40,
    "max_steps_per_episode": 100,
    "max_episodes": 3000,
    "epsilon_max": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.00009,
    "memory_capacity": 10000,
    "g_ap": 28.0,
    "g_p": 3.0,
    "g_bias": 24.0,
    "CP_pole_length_2": 8.0,
    "CP_cart_mass_2": 0.3
}

train_mode = True


if __name__ == "__main__":
    if train_mode:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = Path(f"run_{timestamp}")
        # Create the folder
        folder.mkdir(parents=True, exist_ok=True)

        seed = 124
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        trainer = Trainer(hyperparams, seed, folder)
        rewards_cp, rewards_mc = trainer.train()

        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ----- CP Plot -----
        axes[0].plot(rewards_cp, label="Reward CP", linewidth=2)
        axes[0].set_xlabel("Episode", fontsize=14)
        axes[0].set_ylabel("Reward", fontsize=14)
        axes[0].set_title("Training on CP with Pole Length 0.5", fontsize=16)
        axes[0].grid(True)

        # ----- MC Plot -----
        axes[1].plot(rewards_mc, label="Reward MC", linewidth=2)
        axes[1].set_xlabel("Episode", fontsize=14)
        axes[1].set_ylabel("Reward", fontsize=14)
        axes[1].set_title(f"Training on CP with Pole Length {hyperparams["CP_pole_length_2"]}", fontsize=16)
        axes[1].grid(True)

        plt.tight_layout()
        plot_path = folder / "training_plot.png"
        plt.savefig(plot_path, dpi=300)
        plt.show()
    else:
        trainer = Trainer(hyperparams, seed=None)
        trainer.test("CP_best_model_seed_124_8810.pth")