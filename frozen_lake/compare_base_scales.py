import csv
import time
from pathlib import Path
from statistics import mean

from model_train_test import ModelTrainTest


DEFAULT_HYPER = {
    "train_mode": True,
    "RL_load_path": './4x4_weights/final_weights_3000.pth',
    "save_path": './4x4_weights/final_weights',
    "save_interval": 500,

    "clip_grad_norm": 3,
    "learning_rate": 6e-4,
    "discount_factor": 0.93,
    "batch_size": 32,
    "update_frequency": 10,
    # replace max_episodes when running experiments
    "max_episodes": 50,
    "max_steps": 200,
    "render": False,

    "max_epsilon": 0.999,
    "min_epsilon": 0.01,
    "epsilon_decay": 0.999,

    "memory_capacity": 4000,

    "map_size": 4,
    "num_states": 16,
    "render_fps": 6,
}


def run_experiment(base_scale, episodes=50, out_dir='./compare_results'):
    hp = DEFAULT_HYPER.copy()
    hp['max_episodes'] = episodes
    hp['base_scale'] = base_scale
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    start = time.time()
    mt = ModelTrainTest(hp)
    mt.train()
    duration = time.time() - start

    # metrics
    avg_reward = mean(mt.reward_history) if mt.reward_history else 0.0
    final_reward = mt.reward_history[-1] if mt.reward_history else 0.0
    success_rate = sum(1 for r in mt.reward_history if r > 0) / len(mt.reward_history)

    out = {
        'base_scale': base_scale,
        'episodes': episodes,
        'avg_reward': avg_reward,
        'final_reward': final_reward,
        'success_rate': success_rate,
        'duration_sec': round(duration, 2),
    }

    csv_path = Path(out_dir) / 'results.csv'
    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(out.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(out)

    return out


if __name__ == '__main__':
    # quick smoke test with two base scales
    scales = [1e6, 9e7]
    results = []
    for s in scales:
        print('Running base_scale=', s)
        r = run_experiment(s, episodes=10, out_dir='./compare_results')
        print(r)
        results.append(r)

    print('\nAll runs complete. Results saved to ./compare_results/results.csv')

