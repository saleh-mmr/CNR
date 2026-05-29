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
	"network_size": 50,
	"max_steps_per_episode": 200,
	"max_episodes": 5000,
	"epsilon_max": 1.0,
	"epsilon_min": 0.01,
	"epsilon_decay": 0.00005,
	"memory_capacity": 10000,
	"g_ap": 13.0,
	"g_p": 11.0,
	"shift_parameter": 20,
	"g_bias": 45.0,
	"noise_stddev": 0.05,
	"CP_pole_length_1": 5.5,
	"CP_pole_mass_1": 0.1,
	"CP_pole_length_2": 10,
	"CP_pole_mass_2": 0.2,
	"CP_pole_length_3": 20.5,
	"CP_pole_mass_3": 0.3,
}


if __name__ == "__main__":
    seed = 873
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    folder = Path("three_problems") / f"run_{timestamp}"
    folder.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    trainer = Trainer(hyperparams, seed, folder)
    rewards = trainer.train()

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(rewards, label="Reward", linewidth=4)
    ax.set_xlabel("Episode", fontsize=15)
    ax.set_ylabel("Reward", fontsize=15)
    ax.set_title(
        f"Training on Cartpole Problems with Pole Length "
        f"{hyperparams['CP_pole_length_1']}, "
        f"{hyperparams['CP_pole_length_2']}, "
        f"{hyperparams['CP_pole_length_3']}",
        fontsize=16
    )
    ax.grid(True)

    plt.tight_layout()
    plot_path = folder / "training_plot.png"
    plt.savefig(plot_path, dpi=300)
    plt.show()