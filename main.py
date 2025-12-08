from config import device
from trainer import Trainer

train_mode = False                                  # True = Train RL model, False = Load and Test
render = False                                      # Enable/disable environment visualization
max_episodes = 300 if train_mode else 10

# ---------------- Hyperparameters ----------------
HPARAMS = {
    "RL_load_pth": "weights/best_cartpole.pth",     # Model checkpoint path for testing
    "learning_rate": 3e-4,                          # Optimizer learning rate
    "discount_factor": 0.99,                        # Gamma (future reward discount)
    "batch_size": 64,                               # Number of experiences per learning step
    "max_episodes": max_episodes,                   # number of episode for training or testing
    "max_steps": 500,                               # Episode timeout
    "render": render,                               # Set True to visually inspect
    "epsilon_max": 0.9,                             # Initial exploration rate
    "epsilon_min": 0.05,                            # Minimum allowed epsilon
    "epsilon_decay": 0.003,                         # Exploration decay speed
    "memory_capacity": 10000,                       # Replay buffer size
    "render_fps": 30,                               # Visualization frame rate
}


if __name__ == "__main__":
    trainer = Trainer(HPARAMS)
    if train_mode:
        print(f"Running on device: {device}")
        trainer.train()
    else:
        trainer.test(max_episodes)
