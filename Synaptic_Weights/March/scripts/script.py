import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from matplotlib import pyplot as plt
import random
import numpy as np
import torch
from learning.trainer import Trainer

hyperparams = {
    "discount_factor": 0.99,
    "batch_size": 5,
    "max_episodes": 30,
    "epsilon_max": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.0008,
    "memory_capacity": 100000,
}

train_mode = True


if __name__ == "__main__":

    if train_mode:
        seeds = [49]
        rewards_cp_dic = {}
        rewards_mc_dic = {}
        for seed in seeds:
            print(f"Training with seed: {seed}")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            trainer = Trainer(hyperparams, seed)
            rewards_cp, rewards_mc = trainer.train()
            rewards_cp_dic[seed] = rewards_cp
            rewards_mc_dic[seed] = rewards_mc
        reward_runs_cp = list(rewards_cp_dic.values())
        reward_runs_mc = list(rewards_mc_dic.values())
        mean_rewards_cp = np.mean(reward_runs_cp, axis=0)
        mean_rewards_mc = np.mean(reward_runs_mc, axis=0)
        std_rewards_cp = np.std(reward_runs_cp, axis=0)
        std_rewards_mc = np.std(reward_runs_mc, axis=0)

        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns

        # ----- CP Plot -----
        for seed, rewards in rewards_cp_dic.items():
            axes[0].plot(rewards, label=f"Seed {seed}", alpha=0.7)

        axes[0].plot(mean_rewards_cp, label="Mean Reward CP", linewidth=3)
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")
        axes[0].set_title("DQN Training on CP with Synaptic Weight Controller")
        axes[0].legend()
        axes[0].grid(True)

        # ----- MC Plot -----
        for seed, rewards in rewards_mc_dic.items():
            axes[1].plot(rewards, label=f"Seed {seed}", alpha=0.7)

        axes[1].plot(mean_rewards_mc, label="Mean Reward MC", linewidth=3)
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Reward")
        axes[1].set_title("DQN Training on MC with Synaptic Weight Controller")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    else:
        trainer = Trainer(hyperparams, seed=None)
        trainer.test("MC_best_model_seed_49_7803.pth")