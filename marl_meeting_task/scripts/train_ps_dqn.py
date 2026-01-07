"""
Training script for Parameter-Shared DQN (PS-DQN) on Meeting Gridworld.

Runs training with multiple seeds and collects aggregated results.
Matches IQL training protocol for fair comparison.
"""

import os
import json
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any

from marl_meeting_task.src.env.meeting_gridworld import MeetingGridworldEnv
from marl_meeting_task.src.algos.ps_dqn import PS_DQN
from marl_meeting_task.src.utils.logger import Logger


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Parameters:
    -----------
    seed : int
        Random seed value
    """
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_training(
    seed: int,
    max_episodes: int = 1000,
    log_dir: str = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run training for a single seed.
    
    Parameters:
    -----------
    seed : int
        Random seed for this run
    max_episodes : int
        Maximum number of training episodes
    log_dir : str
        Directory for TensorBoard logs (None = default)
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    Dict[str, Any]
        Training statistics for this seed
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    # Create environment
    env = MeetingGridworldEnv(grid_size=5, n_agents=2, max_steps=50)
    
    # Initialize PS-DQN
    ps_dqn = PS_DQN(
        n_agents=2,
        input_dim=4,
        num_actions=5,
        hidden_dim=64,
        learning_rate=7e-5,
        memory_capacity=10000,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=50000,
        batch_size=32,
        target_update_freq=500,
    )
    
    # Set log directory for this seed
    if log_dir is None:
        log_dir = f"runs/ps_dqn_seed_{seed}"
    else:
        log_dir = f"{log_dir}/seed_{seed}"
    
    # Train
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training with seed: {seed}")
        print(f"{'='*60}\n")
    
    stats = ps_dqn.train(
        env=env,
        max_episodes=max_episodes,
        max_steps=50,
        train_freq=1,
        min_buffer_size=1000,
        verbose=verbose,
        log_dir=log_dir,
        eval_episodes=200,
        env_seed=seed,  # Pass seed for environment resets
    )
    
    # Add seed to stats
    stats['seed'] = seed
    
    return stats


def aggregate_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results across multiple seeds.
    
    Parameters:
    -----------
    all_results : List[Dict[str, Any]]
        List of results dictionaries from each seed
        
    Returns:
    --------
    Dict[str, Any]
        Aggregated statistics (mean, std, min, max)
    """
    n_seeds = len(all_results)
    
    # Extract arrays for aggregation
    all_rewards = [r['episode_rewards'] for r in all_results]
    all_lengths = [r['episode_lengths'] for r in all_results]
    all_successes = [r['episode_successes'] for r in all_results]
    
    # Find maximum episode length (in case runs had different lengths)
    max_episodes = max(len(r) for r in all_rewards)
    
    # Pad arrays to same length (with NaN for missing episodes)
    padded_rewards = []
    padded_lengths = []
    padded_successes = []
    
    for rewards, lengths, successes in zip(all_rewards, all_lengths, all_successes):
        padded_rewards.append(rewards + [np.nan] * (max_episodes - len(rewards)))
        padded_lengths.append(lengths + [np.nan] * (max_episodes - len(lengths)))
        padded_successes.append(successes + [np.nan] * (max_episodes - len(successes)))
    
    # Convert to numpy arrays
    rewards_array = np.array(padded_rewards)
    lengths_array = np.array(padded_lengths)
    successes_array = np.array(padded_successes)
    
    # Compute statistics
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
            'final_success_rate_mean': float(np.nanmean([np.mean(s[-100:]) for s in all_successes])),
            'final_success_rate_std': float(np.nanstd([np.mean(s[-100:]) for s in all_successes])),
            'final_episode_length_mean': float(np.nanmean([np.mean(l[-100:]) for l in all_lengths])),
            'final_episode_length_std': float(np.nanstd([np.mean(l[-100:]) for l in all_lengths])),
            'final_return_mean': float(np.nanmean([np.mean(r[-100:]) for r in all_rewards])),
            'final_return_std': float(np.nanstd([np.mean(r[-100:]) for r in all_rewards])),
        },
    }
    
    return aggregated


def save_results(
    all_results: List[Dict[str, Any]],
    aggregated: Dict[str, Any],
    output_dir: str = "results",
) -> None:
    """
    Save results to JSON files.
    
    Parameters:
    -----------
    all_results : List[Dict[str, Any]]
        Results from each seed
    aggregated : Dict[str, Any]
        Aggregated statistics
    output_dir : str
        Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save individual seed results
    individual_file = os.path.join(output_dir, f"ps_dqn_individual_seeds_{timestamp}.json")
    with open(individual_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = []
        for r in all_results:
            json_r = {
                'seed': r['seed'],
                'total_steps': r['total_steps'],
                'episode_rewards': r['episode_rewards'],
                'episode_lengths': r['episode_lengths'],
                'episode_successes': r['episode_successes'],
            }
            json_results.append(json_r)
        json.dump(json_results, f, indent=2)
    
    # Save aggregated results
    aggregated_file = os.path.join(output_dir, f"ps_dqn_aggregated_{timestamp}.json")
    with open(aggregated_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"\nResults saved:")
    print(f"  Individual seeds: {individual_file}")
    print(f"  Aggregated: {aggregated_file}")


def main():
    """Main training function."""
    # Configuration (matching IQL protocol)
    SEEDS = [2025, 2026, 2027, 2028, 2029]  # Same seeds as IQL for fair comparison
    MAX_EPISODES = 1000
    BASE_LOG_DIR = "runs/ps_dqn_multi_seed"
    OUTPUT_DIR = "results"
    VERBOSE = True
    
    # Initialize logger for training script
    logger = Logger(verbose=VERBOSE, log_dir=None)
    
    logger.summary(
        title="PS-DQN Multi-Seed Training",
        items={
            "Seeds": SEEDS,
            "Max episodes per seed": MAX_EPISODES,
            "TensorBoard logs": BASE_LOG_DIR,
            "Results directory": OUTPUT_DIR,
        }
    )
    
    # Run training for each seed
    all_results = []
    
    for i, seed in enumerate(SEEDS, 1):
        logger.info(f"\n[{i}/{len(SEEDS)}] Starting training with seed {seed}...")
        
        try:
            stats = run_training(
                seed=seed,
                max_episodes=MAX_EPISODES,
                log_dir=BASE_LOG_DIR,
                verbose=VERBOSE,
            )
            all_results.append(stats)
            
            # Print summary for this seed
            final_success_rate = np.mean(stats['episode_successes'][-100:])
            final_avg_length = np.mean(stats['episode_lengths'][-100:])
            final_avg_return = np.mean(stats['episode_rewards'][-100:])
            
            logger.info(f"\n[Seed {seed} Summary]")
            logger.info(f"  Final Success Rate (last 100): {final_success_rate:.2%}")
            logger.info(f"  Final Avg Length (last 100): {final_avg_length:.1f}")
            logger.info(f"  Final Avg Return (last 100): {final_avg_return:.2f}")
            logger.info(f"  Total Steps: {stats['total_steps']}")
            
        except Exception as e:
            logger.info(f"\n[ERROR] Training failed for seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_results) == 0:
        logger.info("\n[ERROR] No successful training runs!")
        return
    
    # Aggregate results
    aggregated = aggregate_results(all_results)
    
    # Print aggregated summary using logger
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
    
    # Save results
    save_results(all_results, aggregated, OUTPUT_DIR)
    
    logger.training_complete(BASE_LOG_DIR)


if __name__ == "__main__":
    main()

