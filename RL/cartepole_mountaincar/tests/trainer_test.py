import os
import sys
from matplotlib import pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from learning.trainer import Trainer


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
rewards = trainer.train()


plt.figure()
plt.plot(rewards)
plt.xlabel("episode")
plt.ylabel("rewards")
plt.title(f"Total rewards per episode (max={max(rewards):.2f})")
plt.grid(True)
plt.show()
