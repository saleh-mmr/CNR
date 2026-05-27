import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from learning.trainer import Trainer


hyperparams = {
	"discount_factor": 0.99,
	"batch_size": 3000,
	"warmup_size": 3000,
	"network_size": 80,
	"max_steps_per_episode": 100,
	"max_episodes": 4000,
	"epsilon_max": 1.0,
	"epsilon_min": 0.01,
	"epsilon_decay": 0.00007,
	"memory_capacity": 10000,
	"g_ap": 18.0,
	"g_p": 15.0,
	"shift_parameter": 20,
	"g_bias": 30.0,
	"noise_stddev": 0.001,
	"CP_pole_length_1": 5.0,
	"CP_pole_mass_1": 1.0,
	"CP_pole_length_2": 10.0,
	"CP_pole_mass_2": 2.0,
	"CP_pole_length_3": 20.0,
	"CP_pole_mass_3": 5.0,
}


def main():
	folder_name = "run_2026-05-27_19-38-33"
	cartpole_selector = 2
	weight_selector = 2
	num_tests = 3

	base_folder = SCRIPT_DIR / "three_problems"
	folder = base_folder / folder_name
	if not folder.exists():
		print(f"Run folder not found: {folder}")
		return

	keyword_by_selector = {0: "MC1", 1: "MC2", 2: "MC3"}
	keyword = keyword_by_selector.get(weight_selector)
	if keyword is None:
		print("Invalid weight_selector. Use 0, 1, or 2.")
		return

	checkpoint_paths = sorted(
		[path for path in folder.iterdir() if path.is_file() and path.suffix == ".pth" and path.name.startswith(f"{keyword}_")],
		key=lambda path: int(path.stem.split("_")[-1]) if path.stem.split("_")[-1].isdigit() else -1,
	)

	if not checkpoint_paths:
		print(f"No checkpoints found in {folder} for prefix {keyword}_")
		return

	trainer = Trainer(hyperparams, seed=None, folder=folder)

	for checkpoint_path in checkpoint_paths:
		print(f"\n=== Testing checkpoint: {checkpoint_path.name} ===")
		test_log = trainer.test(
			model_path=checkpoint_path,
			num_tests=num_tests,
			cartpole=cartpole_selector,
		)

		mean_reward = float(test_log["reward"].mean())
		std_reward = float(test_log["reward"].std(ddof=0))

		print(
			f"Summary for {checkpoint_path.name} | "
			f"mean={mean_reward:.6f} | std={std_reward:.6f}"
		)


if __name__ == "__main__":
	main()
