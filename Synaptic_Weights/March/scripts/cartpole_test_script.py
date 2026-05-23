import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from learning.trainer import Trainer



hyperparams = {
    "discount_factor": 0.99,
    "batch_size": 3000,
    "warmup_size": 3000,
    "network_size": 100,
    "max_steps_per_episode": 200,
    "max_episodes": 9000,
    "epsilon_max": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.00001,
    "memory_capacity": 10000,
    "g_ap": 25.0,
    "g_p": 22.0,
    "shift_parameter": 6,
    "g_bias": 45.0,
    "noise_stddev": 0.01,
    "CP_pole_length_2": 5.0,
    "CP_pole_mass_2": 0.5
}

if __name__ == "__main__":
        folder = "weee"
        weigh_step = 347742
        cartpole = 1
        render = False
        keyword = "CP" if cartpole == 0 else "MC"
        path = f"{folder}/{keyword}_best_model_{weigh_step}.pth"
        num_tests = 1000
        trainer_CP = Trainer(hyperparams, seed=None, folder=folder)
        test_log = trainer_CP.test(model_path=path, num_tests=num_tests, cartpole=cartpole, render=render)
        result_path = f"test_log_{keyword}.csv"
        test_log.to_csv(f"{folder}/{result_path}", index=False)