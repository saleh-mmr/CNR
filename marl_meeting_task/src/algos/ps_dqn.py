import os
import numpy as np
import torch
from typing import Dict, Optional, Any
from marl_meeting_task.src.algos.ps_dqn_agent import PS_DQNAgent
from marl_meeting_task.src.config import device
from marl_meeting_task.src.utils.logger import Logger


class PS_DQN:
    """
    Parameter-Shared Deep Q-Network (PS-DQN) for Multi-Agent Reinforcement Learning.
    
    PS-DQN uses a single shared Q-network for all agents, enabling parameter
    sharing which can improve sample efficiency and generalization.
    
    Key differences from IQL:
    - Single shared Q-network (instead of one per agent)
    - Single shared optimizer
    - Single shared replay buffer (stores per-agent transitions)
    - Each timestep contributes n_agents samples to the buffer
    
    Reference:
    Tan, M. (1993). Multi-agent reinforcement learning: Independent vs.
    cooperative agents. ICML.
    """
    
    def __init__(
        self,
        n_agents: int = 2,
        input_dim: int = 4,  # observation vector: [own_x, own_y, goal_x, goal_y]
        num_actions: int = 5,  # actions: up, down, left, right, stay
        hidden_dim: int = 64,
        learning_rate: float = 7e-5,
        memory_capacity: int = 10000,
        gamma: float = 0.99,  # Discount factor
        epsilon_start: float = 1.0,  # Initial epsilon
        epsilon_end: float = 0.05,  # Final epsilon
        epsilon_decay_steps: int = 50000,  # Steps over which epsilon decays
        batch_size: int = 32,  # Batch size for training
        target_update_freq: int = 500,  # Update target network every N steps
    ):
        """
        Initialize Parameter-Shared DQN.
        
        Parameters:
        -----------
        n_agents : int
            Number of agents (default: 2)
        input_dim : int
            Dimension of observation vector (default: 4)
        num_actions : int
            Number of possible actions (default: 5)
        hidden_dim : int
            Hidden layer dimension for Q-network (default: 64)
        learning_rate : float
            Learning rate for optimizer (default: 1e-3)
        memory_capacity : int
            Capacity of shared replay buffer (default: 10000)
        gamma : float
            Discount factor (default: 0.99)
        epsilon_start : float
            Initial exploration rate (default: 1.0)
        epsilon_end : float
            Final exploration rate (default: 0.05)
        epsilon_decay_steps : int
            Steps over which epsilon decays (default: 50000)
        batch_size : int
            Batch size for training (default: 32)
        target_update_freq : int
            Update target network every N steps (default: 500)
        """
        # Store hyperparameters
        self.n_agents = n_agents
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.memory_capacity = memory_capacity
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Training state
        self.total_steps = 0
        
        # Initialize shared agent component
        self.agent = PS_DQNAgent(
            n_agents=n_agents,
            input_dim=input_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            memory_capacity=memory_capacity,
            gamma=gamma,
            batch_size=batch_size,
        )
        
        # Logger will be initialized in train() method
        self._logger: Optional[Logger] = None
    
    def _print_initialization_summary(self, logger: Logger) -> None:
        """Print initialization summary."""
        logger.info(f"PS-DQN initialized with {self.n_agents} agents sharing parameters")
        logger.info(f"  - Input dimension: {self.input_dim}")
        logger.info(f"  - Number of actions: {self.num_actions}")
        logger.info(f"  - Shared Q-value network: {self.input_dim} -> {self.hidden_dim} -> {self.hidden_dim} -> {self.num_actions}")
        logger.info(f"  - Target network: (updates every {self.target_update_freq} steps)")
        logger.info(f"  - Shared optimizer: Adam (lr={self.learning_rate})")
        logger.info(f"  - Shared replay buffer: capacity={self.memory_capacity}")
        logger.info(f"  - Hyperparameters:")
        logger.info(f"    * Gamma (discount): {self.gamma}")
        logger.info(f"    * Epsilon: {self.epsilon_start} -> {self.epsilon_end} over {self.epsilon_decay_steps} steps")
        logger.info(f"    * Batch size: {self.batch_size}")
    
    # ========================================================================
    # Exploration Schedule
    # ========================================================================
    
    def get_epsilon(self) -> float:
        """
        Compute current epsilon based on linear decay schedule.
        
        Returns:
        --------
        epsilon : float
            Current exploration rate
        """
        if self.total_steps >= self.epsilon_decay_steps:
            return self.epsilon_end
        
        # Linear decay: ε = ε_start - (ε_start - ε_end) * (steps / decay_steps)
        epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (
            self.total_steps / self.epsilon_decay_steps
        )
        return max(epsilon, self.epsilon_end)
    
    # ========================================================================
    # Action Selection
    # ========================================================================
    
    def select_actions(self, obs: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        Select actions for all agents using epsilon-greedy policy with shared network.
        
        Parameters:
        -----------
        obs : Dict[int, np.ndarray]
            Observations keyed by agent_id
            
        Returns:
        --------
        Dict[int, int]
            Actions keyed by agent_id
        """
        epsilon = self.get_epsilon()
        return self.agent.select_actions(obs, epsilon)
    
    # ========================================================================
    # Experience Storage
    # ========================================================================
    
    def store_transitions(
        self,
        obs: Dict[int, np.ndarray],
        actions: Dict[int, int],
        next_obs: Dict[int, np.ndarray],
        reward: float,
        done: bool
    ) -> None:
        """
        Store transitions in shared replay buffer.
        
        Parameters:
        -----------
        obs : Dict[int, np.ndarray]
            Current observations keyed by agent_id
        actions : Dict[int, int]
            Actions taken keyed by agent_id
        next_obs : Dict[int, np.ndarray]
            Next observations keyed by agent_id
        reward : float
            Scalar reward (shared by all agents)
        done : bool
            Whether episode terminated or truncated
        """
        self.agent.store_transitions(obs, actions, next_obs, reward, done)
    
    # ========================================================================
    # Training
    # ========================================================================
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step using shared network.
        
        Returns:
        --------
        Optional[float]
            Training loss if buffer has enough samples, None otherwise
        """
        return self.agent.train_step()
    
    # ========================================================================
    # Evaluation
    # ========================================================================
    
    def evaluate(
        self,
        env,
        n_episodes: int = 20,
        max_steps: int = 50,
    ) -> Dict[str, float]:
        """
        Evaluate the current policy with greedy actions (epsilon=0).
        
        Parameters:
        -----------
        env : MeetingGridworldEnv
            The environment to evaluate on
        n_episodes : int
            Number of evaluation episodes (default: 20)
        max_steps : int
            Maximum steps per episode (default: 50)
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing evaluation metrics:
            - success_rate: fraction of successful episodes
            - avg_episode_length: average episode length
            - avg_return: average cumulative reward
        """
        # Set network to eval mode for inference
        self.agent.q_network.eval()
        
        eval_successes = []
        eval_lengths = []
        eval_returns = []
        
        for episode in range(n_episodes):
            # Use None seed for evaluation to get diverse episodes
            obs, info = env.reset(seed=None)
            episode_reward = 0.0
            episode_terminated = False
            
            for t in range(max_steps):
                # Ensure network remains in eval mode
                if self.agent.q_network.training:
                    self.agent.q_network.eval()
                
                # Greedy action selection (epsilon=0) - use main network
                actions = {}
                for agent_id in range(self.n_agents):
                    with torch.no_grad():
                        # Convert observation to float32 tensor (observations are int64 from env)
                        obs_tensor = torch.as_tensor(
                            obs[agent_id],
                            dtype=torch.float32,
                            device=device
                        ).unsqueeze(0)
                        q_values = self.agent.q_network(obs_tensor)
                        actions[agent_id] = q_values.argmax().item()
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                
                if terminated:
                    episode_terminated = True
                
                obs = next_obs
                episode_reward += reward
                
                if done:
                    break
            
            # Record evaluation episode statistics
            eval_successes.append(1 if episode_terminated else 0)
            eval_lengths.append(t + 1)
            eval_returns.append(episode_reward)
        
        # Set network back to train mode
        self.agent.q_network.train()
        
        # Compute evaluation metrics
        success_rate = np.mean(eval_successes)
        avg_episode_length = np.mean(eval_lengths)
        avg_return = np.mean(eval_returns)
        
        return {
            'success_rate': success_rate,
            'avg_episode_length': avg_episode_length,
            'avg_return': avg_return,
        }
    
    def train(
        self,
        env,
        max_episodes: int,
        max_steps: int = 50,
        train_freq: int = 1,
        min_buffer_size: int = 1000,
        verbose: bool = True,
        log_dir: Optional[str] = "runs/ps_dqn",
        eval_episodes: int = 20,
        env_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Main training loop for Parameter-Shared DQN.
        
        Parameters:
        -----------
        env : MeetingGridworldEnv
            The environment to train on
        max_episodes : int
            Maximum number of episodes to train
        max_steps : int
            Maximum steps per episode (default: 50)
        train_freq : int
            Train every N steps (default: 1)
        min_buffer_size : int
            Minimum buffer size before training starts (default: 1000)
        verbose : bool
            Whether to print training progress (default: True)
        log_dir : Optional[str]
            Directory for TensorBoard logs (default: "runs/ps_dqn")
            If None, TensorBoard logging is disabled
        eval_episodes : int
            Number of episodes to run during evaluation (default: 20)
        env_seed : Optional[int]
            Seed for environment resets (default: None, uses config seed)
            
        Returns:
        --------
        training_stats : dict
            Dictionary containing training statistics
        """
        # Initialize logger
        logger = Logger(verbose=verbose, log_dir=log_dir)
        self._logger = logger
        
        # Print initialization summary
        self._print_initialization_summary(logger)
        
        # Episode statistics
        episode_rewards = []
        episode_lengths = []
        episode_successes = []  # 0 or 1 for each episode
        episode_losses = []
        
        # Moving averages for TensorBoard (window size: 100 episodes)
        window_size = 100
        success_window = []
        length_window = []
        return_window = []
        
        for episode in range(max_episodes):
            # Use seed derived from base seed and episode for reproducibility with diversity
            episode_seed = None if env_seed is None else env_seed + episode
            obs, info = env.reset(seed=episode_seed)
            episode_reward = 0.0
            episode_terminated = False  # Track if episode ended with success
            episode_loss_sum = 0.0
            episode_loss_count = 0
            
            for t in range(max_steps):
                # Select actions for all agents (epsilon-greedy)
                actions = self.select_actions(obs)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                
                # Track if episode ended with success (all agents reached goal)
                if terminated:
                    episode_terminated = True
                
                # Store transitions in shared buffer
                self.store_transitions(obs, actions, next_obs, reward, done)
                
                # Train if buffer is large enough
                if self.total_steps % train_freq == 0:
                    if len(self.agent.replay_memory) >= min_buffer_size:
                        loss = self.train_step()
                        if loss is not None:
                            episode_loss_sum += loss
                            episode_loss_count += 1
                
                # Update target network periodically
                if self.total_steps > 0 and self.total_steps % self.target_update_freq == 0:
                    self.agent.update_target_network()
                
                # Update state
                obs = next_obs
                episode_reward += reward
                self.total_steps += 1
                
                if done:
                    break
            
            # Record episode statistics
            episode_length = t + 1
            episode_success = 1 if episode_terminated else 0
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_successes.append(episode_success)
            
            if episode_loss_count > 0:
                avg_loss = episode_loss_sum / episode_loss_count
                episode_losses.append(avg_loss)
            else:
                episode_losses.append(None)
            
            # Update moving average windows
            success_window.append(episode_success)
            length_window.append(episode_length)
            return_window.append(episode_reward)
            
            # Keep window size fixed
            if len(success_window) > window_size:
                success_window.pop(0)
                length_window.pop(0)
                return_window.pop(0)
            
            # Log to TensorBoard and console
            logger.tensorboard_log_metrics(
                episode=episode,
                success=episode_success,
                length=episode_length,
                return_val=episode_reward,
                loss=episode_losses[-1],
                epsilon=self.get_epsilon(),
            )
            
            # Moving averages (computed over window)
            if len(success_window) >= window_size:
                success_rate = np.mean(success_window)
                avg_episode_length = np.mean(length_window)
                avg_return = np.mean(return_window)
                
                logger.tensorboard_log_moving_averages(
                    episode=episode,
                    success_rate=success_rate,
                    avg_episode_length=avg_episode_length,
                    avg_return=avg_return,
                )
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                success_rate = np.mean(episode_successes[-100:])
                current_epsilon = self.get_epsilon()
                logger.progress(
                    episode=episode + 1,
                    max_episodes=max_episodes,
                    avg_reward=avg_reward,
                    avg_length=avg_length,
                    success_rate=success_rate,
                    epsilon=current_epsilon,
                    total_steps=self.total_steps,
                )
        
        # Run final evaluation after all training episodes
        final_eval_metrics = self.evaluate(env, n_episodes=eval_episodes, max_steps=max_steps)
        
        # Log final evaluation metrics
        logger.tensorboard_log_evaluation(
            episode=max_episodes - 1,
            success_rate=final_eval_metrics['success_rate'],
            avg_episode_length=final_eval_metrics['avg_episode_length'],
            avg_return=final_eval_metrics['avg_return'],
        )
        
        # Print final evaluation results
        logger.evaluation(
            episode=None,
            success_rate=final_eval_metrics['success_rate'],
            avg_episode_length=final_eval_metrics['avg_episode_length'],
            avg_return=final_eval_metrics['avg_return'],
            is_final=True,
        )
        
        # Close logger (closes TensorBoard writer)
        logger.close()
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_successes': episode_successes,
            'episode_losses': episode_losses,
            'total_steps': self.total_steps,
            'final_eval_metrics': final_eval_metrics,
        }
