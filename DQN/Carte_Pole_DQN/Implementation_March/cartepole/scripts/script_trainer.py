import os
import sys
from matplotlib import pyplot as plt
import random
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from learning.trainer import Trainer


hyperparams = {
    "discount_factor": 0.99,
    "batch_size": 64,
    "max_episodes": 2000,
    "max_steps": 200,
    "epsilon_max": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.00008,
    "memory_capacity": 100000,
}

train_mode = False


if __name__ == "__main__":

    if train_mode:

        seeds = [42, 123, 999, 2026]

        rewards_list = {}
        loss_list = {}

        for seed in seeds:

            print(f"Training with seed: {seed}")

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            trainer = Trainer(hyperparams, seed)

            rewards, loss = trainer.train()

            rewards_list[seed] = rewards
            loss_list[seed] = loss

        reward_runs = list(rewards_list.values())

        mean_rewards = np.mean(reward_runs, axis=0)
        std_rewards = np.std(reward_runs, axis=0)

        # Plot results
        plt.figure(figsize=(10,6))

        for seed, rewards in rewards_list.items():
            plt.plot(rewards, label=f"Seed {seed}", alpha=0.7)

        plt.plot(mean_rewards, label="Mean Reward", linewidth=3)

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("DQN Training on CartPole (Multiple Seeds)")
        plt.legend()
        plt.grid(True)

        plt.show()

    else:

        trainer = Trainer(hyperparams, seed=None)
        trainer.test("best_model_seed_2026.pth")


# # Plot 2: Loss
# plt.figure()
# plt.plot(loss)
# plt.xlabel("steps")
# plt.ylabel("Losses")
# plt.title(f"CP Loss per Episode")
# plt.grid(True)
# plt.show()