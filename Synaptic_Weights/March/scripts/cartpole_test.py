import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from learning.trainer import Trainer



hyperparams = {
    "discount_factor": 0.99,
    "batch_size": 3000,
    "warmup_size": 3000,
    "network_size": 100,
    "max_steps_per_episode": 300,
    "max_episodes": 4000,
    "epsilon_max": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.00001,
    "memory_capacity": 10000,
    "g_ap": 25.0,
    "g_p": 22.0,
    "shift_parameter": 6,
    "g_bias": 45.0,
    "noise_stddev": 0.02,
    "CP_pole_length_2": 5.0,
    "CP_pole_mass_2": 0.5
}

if __name__ == "__main__":
        folder = "run_2026-05-25_00-56-52"
        weigh_step = 480256
        cartpole = 1
        correspond_weight = 0
        render = False
        if correspond_weight == 0:
            keyword = "MC_1"
        elif correspond_weight == 1:
            keyword = "MC_2"
        else:
            keyword = "MC_3"
        path = f"{folder}/{keyword}_{weigh_step}.pth"
        num_tests = 1000
        trainer_CP = Trainer(hyperparams, seed=None, folder=folder)
        test_log = trainer_CP.test(model_path=path, num_tests=num_tests, cartpole=cartpole, render=render)
        result_path = f"test_log_{keyword}.csv"
        test_log.to_csv(f"{folder}/{result_path}", index=False)