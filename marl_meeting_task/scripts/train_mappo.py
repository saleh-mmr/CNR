"""
Training script for MAPPO on Meeting Gridworld.

Mirrors the structure and logging behavior of `scripts/train_qmix.py` so
training runs are comparable across algorithms.
"""

import os
import numpy as np
import torch
from typing import Dict, List, Any, Optional
import json
import argparse

from marl_meeting_task.src.env.meeting_gridworld import MeetingGridworldEnv
from marl_meeting_task.src.algos.mappo import MAPPO
from marl_meeting_task.src.utils.logger import Logger


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_training(seed: int, max_updates: int, log_dir: Optional[str], verbose: bool):
    """
    Run MAPPO training for a single seed.

    Returns a dictionary containing per-update statistics and hyperparameters.
    """
    set_seed(seed)

    # Hyperparameters (kept in one dict for easy saving)
    hyperparameters: Dict[str, Any] = {
        'max_updates': max_updates,
        'n_agents': 2,
        'obs_dim': 4,
        'action_dim': 5,
        'hidden_dim': 64,
        'rollout_length': 32,
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'clip_eps': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'ppo_epochs': 4,
        'mini_batch_size': 64,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'share_actor': True,
        'use_centralized_critic': True,
        'max_steps_per_episode': 50,
    }

    # Create environment
    env = MeetingGridworldEnv(grid_size=5, n_agents=hyperparameters['n_agents'], max_steps=hyperparameters['max_steps_per_episode'])

    # Initialize MAPPO
    trainer = MAPPO(
        n_agents=hyperparameters['n_agents'],
        obs_dim=hyperparameters['obs_dim'],
        action_dim=hyperparameters['action_dim'],
        config=hyperparameters,
    )

    # Logging
    if log_dir is None:
        log_dir = f"runs/mappo_seed_{seed}"
    else:
        log_dir = f"{log_dir}/seed_{seed}"

    if verbose:
        print(f"\n{'='*60}")
        print(f"MAPPO training with seed: {seed}")
        print(f"{'='*60}\n")

    # Stats to return
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_successes: List[int] = []
    episode_losses: List[Dict[str, float]] = []
    total_steps = 0

    for update in range(max_updates):
        # Collect a rollout and compute simple rollout-level stats
        last_values = trainer.collect_rollouts(env)

        # Buffer stores rewards per step (accessible before update clears it)
        rollout_rewards = trainer.buffer.rewards  # list of floats
        rollout_length = len(rollout_rewards)
        rollout_reward_sum = float(sum(rollout_rewards))
        rollout_success = int(any(r >= 9.0 for r in rollout_rewards))

        # Update the policy
        losses = trainer.update(last_values)

        # Accumulate stats
        episode_rewards.append(rollout_reward_sum)
        episode_lengths.append(rollout_length)
        episode_successes.append(rollout_success)
        episode_losses.append(losses)
        total_steps += rollout_length

        if verbose:
            print(f"Update {update:3d}/{max_updates} | reward={rollout_reward_sum:.2f} | length={rollout_length} | success={rollout_success} | losses={losses}")

    # Minimal final evaluation metrics placeholder
    final_eval_metrics = {
        'note': 'No separate evaluation loop implemented',
        'avg_reward_per_update': float(np.mean(episode_rewards)) if episode_rewards else 0.0,
    }

    stats = {
        'seed': seed,
        'total_steps': total_steps,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_successes': episode_successes,
        'episode_losses': episode_losses,
        'final_eval_metrics': final_eval_metrics,
        'hyperparameters': hyperparameters,
    }

    return stats


# Reuse aggregation logic from train_qmix.py (works with per-update arrays just fine)
def aggregate_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_seeds = len(all_results)
    all_rewards = [r['episode_rewards'] for r in all_results]
    all_lengths = [r['episode_lengths'] for r in all_results]
    all_successes = [r['episode_successes'] for r in all_results]
    max_updates = max(len(r) for r in all_rewards)

    padded_rewards = []
    padded_lengths = []
    padded_successes = []

    for rewards, lengths, successes in zip(all_rewards, all_lengths, all_successes):
        padded_rewards.append(rewards + [np.nan] * (max_updates - len(rewards)))
        padded_lengths.append(lengths + [np.nan] * (max_updates - len(lengths)))
        padded_successes.append(successes + [np.nan] * (max_updates - len(successes)))

    rewards_array = np.array(padded_rewards)
    lengths_array = np.array(padded_lengths)
    successes_array = np.array(padded_successes)

    aggregated = {
        'n_seeds': n_seeds,
        'seeds': [r['seed'] for r in all_results],
        'episode_rewards': {
            'mean': np.nanmean(rewards_array, axis=0).tolist(),
            'std': np.nanstd(rewards_array, axis=0).tolist(),
            'min': np.nanmin(rewards_array, axis=0).tolist(),
            'max': np.nanmax(rewards_array, axis=0).tolist(),
        },
        'episode_lengths': {
            'mean': np.nanmean(lengths_array, axis=0).tolist(),
            'std': np.nanstd(lengths_array, axis=0).tolist(),
            'min': np.nanmin(lengths_array, axis=0).tolist(),
            'max': np.nanmax(lengths_array, axis=0).tolist(),
        },
        'episode_successes': {
            'mean': np.nanmean(successes_array, axis=0).tolist(),
            'std': np.nanstd(successes_array, axis=0).tolist(),
        },
        'final_metrics': {
            'final_success_rate_mean': float(np.nanmean([np.mean(s[-10:]) for s in all_successes])),
            'final_success_rate_std': float(np.nanstd([np.mean(s[-10:]) for s in all_successes])),
            'final_episode_length_mean': float(np.nanmean([np.mean(l[-10:]) for l in all_lengths])),
            'final_episode_length_std': float(np.nanstd([np.mean(l[-10:]) for l in all_lengths])),
            'final_return_mean': float(np.nanmean([np.mean(r[-10:]) for r in all_rewards])),
            'final_return_std': float(np.nanstd([np.mean(r[-10:]) for r in all_rewards])),
        },
    }

    return aggregated


def main():
    SEEDS = [2025, 2026, 2027, 2028, 2029]
    MAX_EPISODES = 800
    BASE_LOG_DIR = "runs/mappo_multi_seed"
    OUTPUT_DIR = "results"
    VERBOSE = True

    logger = Logger(verbose=VERBOSE, log_dir=None)

    logger.summary(
        title="MAPPO Multi-Seed Training",
        items={
            "Seeds": SEEDS,
            "Max updates per seed": MAX_EPISODES,
            "TensorBoard logs": BASE_LOG_DIR,
            "Results directory": OUTPUT_DIR,
        }
    )

    run_dir = logger.create_run_dir('mappo', base_dir=OUTPUT_DIR)

    all_results = []

    for i, seed in enumerate(SEEDS, 1):
        logger.info(f"\n[{i}/{len(SEEDS)}] Starting MAPPO training with seed {seed}...")
        try:
            stats = run_training(seed=seed, max_updates=MAX_EPISODES, log_dir=BASE_LOG_DIR, verbose=VERBOSE)
            all_results.append(stats)

            # Summarize
            final_success_rate = np.mean(stats['episode_successes'][-10:])
            final_avg_length = np.mean(stats['episode_lengths'][-10:])
            final_avg_return = np.mean(stats['episode_rewards'][-10:])

            logger.info(f"\n[Seed {seed} Summary]")
            logger.info(f"  Final Success Rate (last 10 updates): {final_success_rate:.2%}")
            logger.info(f"  Final Avg Length (last 10 updates): {final_avg_length:.1f}")
            logger.info(f"  Final Avg Return (last 10 updates): {final_avg_return:.2f}")
            logger.info(f"  Total Steps: {stats['total_steps']}")

            # Save per-seed results in run dir
            results_data = {
                'seed': stats['seed'],
                'total_steps': stats['total_steps'],
                'episode_rewards': stats['episode_rewards'],
                'episode_losses': stats['episode_losses'],
                'final_eval_metrics': stats['final_eval_metrics'],
            }

            logger.save_seed_result_in_run(
                run_dir=run_dir,
                seed=seed,
                hyperparameters=stats['hyperparameters'],
                results=results_data,
            )

        except Exception as e:
            logger.info(f"\n[ERROR] Training failed for seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_results) == 0:
        logger.info("\n[ERROR] No successful training runs!")
        return

    aggregated = aggregate_results(all_results)

    aggregated_save = {
        'hyperparameters': all_results[0]['hyperparameters'] if all_results else {},
        'aggregated': aggregated,
    }
    logger.save_aggregated_run(run_dir, aggregated_save)

    logger.aggregated_results(
        n_seeds=aggregated['n_seeds'],
        seeds=aggregated['seeds'],
        success_rate_mean=aggregated['final_metrics']['final_success_rate_mean'],
        success_rate_std=aggregated['final_metrics']['final_success_rate_std'],
        episode_length_mean=aggregated['final_metrics']['final_episode_length_mean'],
        episode_length_std=aggregated['final_metrics']['final_episode_length_std'],
        return_mean=aggregated['final_metrics']['final_return_mean'],
        return_std=aggregated['final_metrics']['final_return_std'],
    )

    logger.training_complete(BASE_LOG_DIR)


if __name__ == '__main__':
    main()

