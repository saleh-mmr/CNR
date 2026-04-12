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
    "batch_size": 2000,
    "warmup_size": 2000,
    "network_size": 40,
    "max_steps_per_episode": 100,
    "max_episodes": 300,
    "epsilon_max": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.0007,
    "memory_capacity": 100000,
    "G_ap": 28.0,
    "G_p:": 3.0,
    "G_bias": 24.0

}

train_mode = True


if __name__ == "__main__":

    # if train_mode:
    #     seeds = [124]  # Example seeds for reproducibility
    #     rewards_cp_dic = {}
    #     rewards_mc_dic = {}
    #     for seed in seeds:
    #         print(f"Training with seed: {seed}")
    #         random.seed(seed)
    #         np.random.seed(seed)
    #         torch.manual_seed(seed)
    #         trainer = Trainer(hyperparams, seed)
    #         rewards_cp, rewards_mc = trainer.train()
    #         rewards_cp_dic[seed] = rewards_cp
    #         rewards_mc_dic[seed] = rewards_mc
    #     reward_runs_cp = list(rewards_cp_dic.values())
    #     reward_runs_mc = list(rewards_mc_dic.values())
    #     mean_rewards_cp = np.mean(reward_runs_cp, axis=0)
    #     mean_rewards_mc = np.mean(reward_runs_mc, axis=0)
    #     std_rewards_cp = np.std(reward_runs_cp, axis=0)
    #     std_rewards_mc = np.std(reward_runs_mc, axis=0)
    #
    #     # Plot results
    #     fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns
    #
    #     # ----- CP Plot -----
    #     for seed, rewards in rewards_cp_dic.items():
    #         axes[0].plot(rewards, label=f"Seed {seed}", alpha=0.7)
    #
    #     axes[0].plot(mean_rewards_cp, label="Mean Reward CP", linewidth=3)
    #     axes[0].set_xlabel("Episode")
    #     axes[0].set_ylabel("Reward")
    #     axes[0].set_title("DQN Training on CP with Synaptic Weight Controller")
    #     axes[0].grid(True)
    #
    #     # ----- MC Plot -----
    #     for seed, rewards in rewards_mc_dic.items():
    #         axes[1].plot(rewards, label=f"Seed {seed}", alpha=0.7)
    #
    #     axes[1].plot(mean_rewards_mc, label="Mean Reward MC", linewidth=3)
    #     axes[1].set_xlabel("Episode")
    #     axes[1].set_ylabel("Reward")
    #     axes[1].set_title("DQN Training on MC with Synaptic Weight Controller")
    #     axes[1].legend()
    #     axes[1].grid(True)
    #
    #     plt.tight_layout()
    #     plt.show()
    if train_mode:
        seed = 124  # single seed
        print(f"Training with seed: {seed}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        trainer = Trainer(hyperparams, seed)
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
        axes[1].set_title("Training on CP with Pole Length 0.7", fontsize=16)
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
    else:
        trainer = Trainer(hyperparams, seed=None)
        trainer.test("CP_best_model_seed_124_8810.pth")