import os
import sys
from matplotlib import pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from learning.trainer import Trainer


hyperparams = {
    "discount_factor": 0.99,
    "batch_size": 64,
    "max_episodes": 4000,
    "max_steps": 200,
    "epsilon_max": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.0001,
    "memory_capacity": 100000,
}

controller_hyperparams = {
    "a": 1.566e-8,
    "b": 0.350e-8,
    "c": 1e6,
    "g_threshold": 0.350e-8,
    "sigma_pulse_noise": 0.0,
    "scaling_factor": 8e8,
    "n_problem": 1,
}
trainer = Trainer(hyperparams, controller_hyperparams)
rewards = trainer.train()
loss = trainer.agent.loss_history


# Plot 1: CP rewards
plt.figure()
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Rewards")
plt.title(f"CP Rewards per Episode (max={max(rewards):.2f})")
plt.grid(True)
plt.show()


# Plot 2: Loss
plt.figure()
plt.plot(loss)
plt.xlabel("Episode")
plt.ylabel("Losses")
plt.title(f"CP Loss per Episode")
plt.grid(True)
plt.show()