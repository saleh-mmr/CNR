"""
Logger class for handling all logging responsibilities in the MARL project.
Handles both console logging and TensorBoard logging.
"""

import os
from typing import Optional, Dict, Any
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Centralized logger for console and TensorBoard logging.
    
    This class handles all logging responsibilities:
    - Console output (info, debug, progress, evaluation results)
    - TensorBoard metrics logging
    - Configuration (verbose mode, log directory)
    """
    
    def __init__(
        self,
        verbose: bool = True,
        log_dir: Optional[str] = None,
    ):
        """
        Initialize the logger.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print to console (default: True)
        log_dir : Optional[str]
            Directory for TensorBoard logs (default: None)
            If None, TensorBoard logging is disabled
        """
        self.verbose = verbose
        self.log_dir = log_dir
        self.writer: Optional[SummaryWriter] = None
        
        # Initialize TensorBoard writer if log_dir is provided
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)
    
    def info(self, message: str) -> None:
        """Log an info message to console."""
        if self.verbose:
            print(message)
    
    def debug(self, message: str) -> None:
        """Log a debug message to console."""
        if self.verbose:
            print(f"[DEBUG] {message}")
    
    def separator(self, char: str = "=", length: int = 60) -> None:
        """Print a separator line."""
        if self.verbose:
            print(char * length)
    
    def blank_line(self) -> None:
        """Print a blank line."""
        if self.verbose:
            print()
    
    def progress(
        self,
        episode: int,
        max_episodes: int,
        avg_reward: float,
        avg_length: float,
        success_rate: float,
        epsilon: float,
        total_steps: int,
    ) -> None:
        """
        Log training progress.
        
        Parameters:
        -----------
        episode : int
            Current episode number
        max_episodes : int
            Maximum number of episodes
        avg_reward : float
            Average reward over last 100 episodes
        avg_length : float
            Average episode length over last 100 episodes
        success_rate : float
            Success rate over last 100 episodes
        epsilon : float
            Current epsilon value
        total_steps : int
            Total training steps
        """
        if self.verbose:
            print(f"Episode {episode}/{max_episodes} | "
                  f"Avg Reward (last 100): {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Success Rate: {success_rate:.2%} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Total Steps: {total_steps}")
    
    def evaluation(
        self,
        episode: Optional[int],
        success_rate: float,
        avg_episode_length: float,
        avg_return: float,
        is_final: bool = False,
    ) -> None:
        """
        Log evaluation results.
        
        Parameters:
        -----------
        episode : Optional[int]
            Episode number (None for final evaluation)
        success_rate : float
            Success rate
        avg_episode_length : float
            Average episode length
        avg_return : float
            Average return
        is_final : bool
            Whether this is the final evaluation (default: False)
        """
        if not self.verbose:
            return
        
        if is_final:
            self.separator()
            self.info("Running final evaluation...")
            self.separator()
            self.blank_line()
            self.info(f"[Final Evaluation] "
                      f"Success Rate: {success_rate:.2%} | "
                      f"Avg Length: {avg_episode_length:.1f} | "
                      f"Avg Return: {avg_return:.2f}")
        else:
            self.blank_line()
            self.info(f"[Evaluation at Episode {episode}] "
                      f"Success Rate: {success_rate:.2%} | "
                      f"Avg Length: {avg_episode_length:.1f} | "
                      f"Avg Return: {avg_return:.2f}")
            self.blank_line()
    
    def tensorboard_log_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int,
    ) -> None:
        """
        Log a scalar value to TensorBoard.
        
        Parameters:
        -----------
        tag : str
            Tag for the scalar (e.g., 'episode/success')
        scalar_value : float
            Scalar value to log
        global_step : int
            Global step number
        """
        if self.writer is not None:
            self.writer.add_scalar(tag, scalar_value, global_step)
    
    def tensorboard_log_metrics(
        self,
        episode: int,
        success: int,
        length: int,
        return_val: float,
        loss: Optional[float] = None,
        epsilon: Optional[float] = None,
    ) -> None:
        """
        Log episode metrics to TensorBoard.
        
        Parameters:
        -----------
        episode : int
            Episode number
        success : int
            Success (0 or 1)
        length : int
            Episode length
        return_val : float
            Episode return
        loss : Optional[float]
            Training loss (default: None)
        epsilon : Optional[float]
            Current epsilon (default: None)
        """
        if self.writer is None:
            return
        
        self.tensorboard_log_scalar('episode/success', success, episode)
        self.tensorboard_log_scalar('episode/length', length, episode)
        self.tensorboard_log_scalar('episode/return', return_val, episode)
        
        if loss is not None:
            self.tensorboard_log_scalar('loss/training', loss, episode)
        
        if epsilon is not None:
            self.tensorboard_log_scalar('exploration/epsilon', epsilon, episode)
    
    def tensorboard_log_moving_averages(
        self,
        episode: int,
        success_rate: float,
        avg_episode_length: float,
        avg_return: float,
    ) -> None:
        """
        Log moving averages to TensorBoard.
        
        Parameters:
        -----------
        episode : int
            Episode number
        success_rate : float
            Success rate over window
        avg_episode_length : float
            Average episode length over window
        avg_return : float
            Average return over window
        """
        if self.writer is None:
            return
        
        self.tensorboard_log_scalar('metrics/success_rate', success_rate, episode)
        self.tensorboard_log_scalar('metrics/episode_length', avg_episode_length, episode)
        self.tensorboard_log_scalar('metrics/return', avg_return, episode)
    
    def tensorboard_log_evaluation(
        self,
        episode: int,
        success_rate: float,
        avg_episode_length: float,
        avg_return: float,
    ) -> None:
        """
        Log evaluation metrics to TensorBoard.
        
        Parameters:
        -----------
        episode : int
            Episode number
        success_rate : float
            Evaluation success rate
        avg_episode_length : float
            Average episode length
        avg_return : float
            Average return
        """
        if self.writer is None:
            return
        
        self.tensorboard_log_scalar('eval/success_rate', success_rate, episode)
        self.tensorboard_log_scalar('eval/episode_length', avg_episode_length, episode)
        self.tensorboard_log_scalar('eval/return', avg_return, episode)
    
    def tensorboard_log_diagnostic(
        self,
        step: int,
        loss: float,
        q_tot: float,
        target: float,
        agent_qs: list,
    ) -> None:
        """
        Log diagnostic information to TensorBoard (e.g., for QMIX).
        
        Parameters:
        -----------
        step : int
            Training step
        loss : float
            Loss value
        q_tot : float
            Total Q value
        target : float
            Target Q value
        agent_qs : list
            List of agent Q values
        """
        if self.writer is None:
            return
        
        self.tensorboard_log_scalar('diagnostic/loss', loss, step)
        self.tensorboard_log_scalar('diagnostic/q_tot', q_tot, step)
        self.tensorboard_log_scalar('diagnostic/target', target, step)
        
        for i, q_val in enumerate(agent_qs):
            self.tensorboard_log_scalar(f'diagnostic/agent_{i}_q', q_val, step)
    
    def summary(
        self,
        title: str,
        items: Dict[str, Any],
    ) -> None:
        """
        Print a formatted summary with title and key-value pairs.
        
        Parameters:
        -----------
        title : str
            Title of the summary
        items : Dict[str, Any]
            Dictionary of key-value pairs to print
        """
        if not self.verbose:
            return
        
        self.blank_line()
        self.separator()
        self.info(title)
        self.separator()
        self.blank_line()
        
        for key, value in items.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.2f}")
            elif isinstance(value, (list, tuple)):
                self.info(f"  {key}: {value}")
            else:
                self.info(f"  {key}: {value}")
    
    def aggregated_results(
        self,
        n_seeds: int,
        seeds: list,
        success_rate_mean: float,
        success_rate_std: float,
        episode_length_mean: float,
        episode_length_std: float,
        return_mean: float,
        return_std: float,
    ) -> None:
        """
        Print aggregated results across multiple seeds.
        
        Parameters:
        -----------
        n_seeds : int
            Number of seeds
        seeds : list
            List of seed values
        success_rate_mean : float
            Mean success rate
        success_rate_std : float
            Std success rate
        episode_length_mean : float
            Mean episode length
        episode_length_std : float
            Std episode length
        return_mean : float
            Mean return
        return_std : float
            Std return
        """
        if not self.verbose:
            return
        
        self.blank_line()
        self.separator()
        self.info("Aggregating results across seeds...")
        self.separator()
        self.blank_line()
        
        self.info("Aggregated Results (across all seeds):")
        self.info(f"  Number of seeds: {n_seeds}")
        self.info(f"  Seeds used: {seeds}")
        self.blank_line()
        self.info("  Final Metrics (last 100 episodes, mean ± std):")
        self.info(f"    Success Rate: {success_rate_mean:.2%} ± {success_rate_std:.2%}")
        self.info(f"    Episode Length: {episode_length_mean:.1f} ± {episode_length_std:.1f}")
        self.info(f"    Return: {return_mean:.2f} ± {return_std:.2f}")
    
    def training_complete(self, log_dir: str) -> None:
        """
        Print training completion message with TensorBoard instructions.
        
        Parameters:
        -----------
        log_dir : str
            Directory where TensorBoard logs are saved
        """
        if not self.verbose:
            return
        
        self.blank_line()
        self.separator()
        self.info("Training completed!")
        self.separator()
        self.blank_line()
        self.info("To view TensorBoard logs:")
        self.info(f"  tensorboard --logdir {log_dir}")
    
    def close(self) -> None:
        """Close the TensorBoard writer and print log directory info."""
        if self.writer is not None:
            self.writer.close()
            if self.verbose:
                self.blank_line()
                self.info(f"TensorBoard logs saved to: {self.log_dir}")
                self.info(f"View with: tensorboard --logdir {self.log_dir}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically close writer."""
        self.close()

