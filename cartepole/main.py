from config import device
from trainer import Trainer

train_mode = True                                  # True = Train RL model, False = Load and Test
render = False                                     # Enable/disable environment visualization
max_episodes = 1000 if train_mode else 10

# ---------------- Hyperparameters ----------------
HPARAMS = {
    "RL_load_pth": "weights/best_cartpole.pth",     # Model checkpoint path for testing
    "learning_rate": 1e-5,                          # (not used for magnitude in Manhattan mode)
    "discount_factor": 0.999,                        # Gamma (future reward discount)
    "batch_size": 16,                               # Number of experiences per learning step
    "max_episodes": max_episodes,                   # number of episode for training or testing
    "max_steps": 500,                               # Episode timeout
    "render": render,                               # Set True to visually inspect
    "epsilon_max": 1.0,                             # Initial exploration rate
    "epsilon_min": 0.00,                            # Minimum allowed epsilon
    "epsilon_decay": 0.0005,                         # Exploration decay speed
    "memory_capacity": 50000,                       # Replay buffer size
    "render_fps": 30,                               # Visualization frame rate
    "weight_datafile_path": "conductance/datafile_V2.csv",         # path to your CSV file
    # Target network parameters
    # "target_update_freq": 200,                     # How many global steps between hard target-network updates
    # "use_soft_update": True,                      # If True, use soft (polyak) updates every step instead of periodic hard copy
    # "target_tau": 0.005,                          # Tau for soft update: target = tau*online + (1-tau)*target
 }


if __name__ == "__main__":
    trainer = Trainer(HPARAMS)
    if train_mode:
        print(f"Running on device: {device}")
        trainer.train()
    else:
        trainer.test(max_episodes)