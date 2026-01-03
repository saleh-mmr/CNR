import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Any
from torch.utils.tensorboard import SummaryWriter

from marl_meeting_task.src.algos.qmix_agent import QMIXAgent
from marl_meeting_task.src.utils.mixing_network import MixingNetwork
from marl_meeting_task.src.utils.qmix_replay_memory import QMIXReplayMemory
from marl_meeting_task.src.config import device


class QMIX:
    """
    QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL.
    
    QMIX uses:
    - Individual agent Q-networks (decentralized execution)
    - Mixing network (centralized training)
    - Centralized replay buffer with global state
    
    Key property: Q_tot is monotonic in each Q_i, enabling decentralized
    execution while maintaining centralized training benefits.
    
    Reference:
    Rashid, T., et al. (2018). QMIX: Monotonic Value Function Factorisation
    for Deep Multi-Agent Reinforcement Learning. ICML.
    """
    
    def __init__(
        self,
        n_agents: int = 2,
        input_dim: int = 6,  # Local observation dimension
        state_dim: int = 6,  # Global state dimension: [a1_x, a1_y, a2_x, a2_y, g_x, g_y]
        num_actions: int = 5,
        hidden_dim: int = 64,
        mixing_hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        memory_capacity: int = 10000,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50000,
        batch_size: int = 32,
        target_update_freq: int = 500,
    ):
        """
        Initialize QMIX.
        
        Parameters:
        -----------
        n_agents : int
            Number of agents (default: 2)
        input_dim : int
            Dimension of local observation (default: 6)
        state_dim : int
            Dimension of global state (default: 6)
        num_actions : int
            Number of possible actions (default: 5)
        hidden_dim : int
            Hidden dimension for agent Q-networks (default: 64)
        mixing_hidden_dim : int
            Hidden dimension for mixing network (default: 64)
        learning_rate : float
            Learning rate (default: 1e-3)
        memory_capacity : int
            Capacity of centralized replay buffer (default: 10000)
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
            Update target networks every N steps (default: 500)
        """
        # Store hyperparameters
        self.n_agents = n_agents
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.mixing_hidden_dim = mixing_hidden_dim
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
        
        # Initialize agent Q-networks (one per agent)
        self.agents: Dict[int, QMIXAgent] = {}
        for agent_id in range(n_agents):
            self.agents[agent_id] = QMIXAgent(
                agent_id=agent_id,
                input_dim=input_dim,
                num_actions=num_actions,
                hidden_dim=hidden_dim,
            )
        
        # Mixing network (main)
        self.mixing_network = MixingNetwork(
            n_agents=n_agents,
            state_dim=state_dim,
            mixing_hidden_dim=mixing_hidden_dim
        ).to(device)
        
        # Mixing network (target)
        self.target_mixing_network = MixingNetwork(
            n_agents=n_agents,
            state_dim=state_dim,
            mixing_hidden_dim=mixing_hidden_dim
        ).to(device)
        self.update_target_networks()  # Initialize with same weights
        
        # Combined optimizer for all networks
        all_params = list(self.mixing_network.parameters())
        for agent in self.agents.values():
            all_params.extend(list(agent.q_network.parameters()))
        
        self.optimizer = optim.Adam(all_params, lr=learning_rate)
        
        # Centralized replay buffer
        self.replay_memory = QMIXReplayMemory(capacity=memory_capacity)
        
        self._print_initialization_summary()
    
    def _print_initialization_summary(self) -> None:
        """Print initialization summary."""
        print(f"QMIX initialized with {self.n_agents} agents")
        print(f"  - Local observation dimension: {self.input_dim}")
        print(f"  - Global state dimension: {self.state_dim}")
        print(f"  - Number of actions: {self.num_actions}")
        print(f"  - Agent Q-networks: {self.input_dim} -> {self.hidden_dim} -> {self.hidden_dim} -> {self.num_actions}")
        print(f"  - Mixing network: combines {self.n_agents} Q-values using state (hidden_dim={self.mixing_hidden_dim})")
        print(f"  - Target networks: (update every {self.target_update_freq} steps)")
        print(f"  - Optimizer: Adam (lr={self.learning_rate})")
        print(f"  - Centralized replay buffer: capacity={self.memory_capacity}")
        print(f"  - Hyperparameters:")
        print(f"    * Gamma (discount): {self.gamma}")
        print(f"    * Epsilon: {self.epsilon_start} -> {self.epsilon_end} over {self.epsilon_decay_steps} steps")
        print(f"    * Batch size: {self.batch_size}")
    
    # ========================================================================
    # Network Management
    # ========================================================================
    
    def update_target_networks(self) -> None:
        """Copy weights from main networks to target networks."""
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        for agent in self.agents.values():
            agent.update_target_network()
    
    # ========================================================================
    # Helper: Extract Global State
    # ========================================================================
    
    def _get_global_state(self, env) -> np.ndarray:
        """
        Extract global state from environment.
        
        Global state: [a1_x, a1_y, a2_x, a2_y, g_x, g_y]
        
        Parameters:
        -----------
        env : MeetingGridworldEnv
            The environment
            
        Returns:
        --------
        np.ndarray
            Global state vector
        """
        state = []
        for agent_id in range(self.n_agents):
            x, y = env.agent_pos[agent_id]
            state.extend([x, y])
        state.extend(env.goal_pos)
        return np.array(state, dtype=np.float32)
    
    # ========================================================================
    # Exploration Schedule
    # ========================================================================
    
    def get_epsilon(self) -> float:
        """
        Compute current epsilon based on linear decay schedule.
        
        Returns:
        --------
        float
            Current exploration rate
        """
        if self.total_steps >= self.epsilon_decay_steps:
            return self.epsilon_end
        
        epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (
            self.total_steps / self.epsilon_decay_steps
        )
        return max(epsilon, self.epsilon_end)
    
    # ========================================================================
    # Action Selection (Decentralized)
    # ========================================================================
    
    def select_actions(self, obs: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        Select actions for all agents (decentralized execution).
        
        Each agent selects action based on its local observation.
        
        Parameters:
        -----------
        obs : Dict[int, np.ndarray]
            Local observations keyed by agent_id
            
        Returns:
        --------
        Dict[int, int]
            Actions keyed by agent_id
        """
        epsilon = self.get_epsilon()
        actions = {}
        for agent_id in range(self.n_agents):
            actions[agent_id] = self.agents[agent_id].select_action(obs[agent_id], epsilon)
        return actions
    
    # ========================================================================
    # Experience Storage
    # ========================================================================
    
    def store_transition(
        self,
        state: np.ndarray,
        obs: Dict[int, np.ndarray],
        actions: Dict[int, int],
        reward: float,
        next_state: np.ndarray,
        next_obs: Dict[int, np.ndarray],
        done: bool
    ) -> None:
        """
        Store transition in centralized replay buffer.
        
        Parameters:
        -----------
        state : np.ndarray
            Global state s
        obs : Dict[int, np.ndarray]
            Local observations o_i
        actions : Dict[int, int]
            Actions a_i
        reward : float
            Joint reward r
        next_state : np.ndarray
            Next global state s_next
        next_obs : Dict[int, np.ndarray]
            Next local observations o_i_next
        done : bool
            Whether episode ended
        """
        self.replay_memory.store(
            state=state,
            observations=obs,
            actions=actions,
            reward=reward,
            next_state=next_state,
            next_observations=next_obs,
            done=done
        )
    
    # ========================================================================
    # Training
    # ========================================================================
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Computes Q_tot loss and updates all networks.
        
        Returns:
        --------
        Optional[float]
            Training loss if buffer has enough samples, None otherwise
        """
        if len(self.replay_memory) < self.batch_size:
            return None
        
        # Sample batch
        states, observations, actions, rewards, next_states, next_observations, dones = \
            self.replay_memory.sample(self.batch_size, self.n_agents)
        
        # Compute current Q_tot(s, a_1, a_2, ...)
        # Get Q_i(o_i, a_i) for each agent
        agent_qs = []
        for agent_id in range(self.n_agents):
            q_values = self.agents[agent_id].q_network(observations[agent_id])  # [batch_size, num_actions]
            # Select Q-value for taken action
            agent_q = q_values.gather(1, actions[agent_id].unsqueeze(1)).squeeze(1)  # [batch_size]
            agent_qs.append(agent_q)
        
        # Stack: [batch_size, n_agents]
        agent_qs = torch.stack(agent_qs, dim=1)
        
        # Mix to get Q_tot
        q_tot = self.mixing_network(agent_qs, states)  # [batch_size, 1]
        q_tot = q_tot.squeeze(1)  # [batch_size]
        
        # Compute target Q_tot(s_next, a_1', a_2', ...)
        # Double-Q: Select actions using online networks, evaluate with target networks
        with torch.no_grad():
            # Step 1: Select greedy actions using online (main) networks
            next_actions = []
            for agent_id in range(self.n_agents):
                next_q_values_online = self.agents[agent_id].q_network(next_observations[agent_id])  # [batch_size, num_actions]
                next_action = next_q_values_online.argmax(1)  # [batch_size] - greedy action from online network
                next_actions.append(next_action)
            
            # Step 2: Evaluate selected actions using target networks
            next_agent_qs = []
            for agent_id in range(self.n_agents):
                next_q_values_target = self.agents[agent_id].target_network(next_observations[agent_id])  # [batch_size, num_actions]
                # Evaluate the action selected by online network using target network
                next_agent_q = next_q_values_target.gather(1, next_actions[agent_id].unsqueeze(1)).squeeze(1)  # [batch_size]
                next_agent_qs.append(next_agent_q)
            
            # Stack: [batch_size, n_agents]
            next_agent_qs = torch.stack(next_agent_qs, dim=1)
            
            # Mix to get Q_tot_target
            q_tot_target = self.target_mixing_network(next_agent_qs, next_states)  # [batch_size, 1]
            q_tot_target = q_tot_target.squeeze(1)  # [batch_size]
            
            # Bellman target
            target = rewards + self.gamma * q_tot_target * (~dones)
        
        # Compute loss
        loss = nn.MSELoss()(q_tot, target)
        
        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability - clip all parameters (agent networks + mixing network)
        all_params = []
        for param_group in self.optimizer.param_groups:
            all_params.extend(param_group['params'])
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
        self.optimizer.step()
        
        return loss.item()
    
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
            Dictionary containing evaluation metrics
        """
        eval_successes = []
        eval_lengths = []
        eval_returns = []
        
        for episode in range(n_episodes):
            obs, info = env.reset(seed=None)
            episode_reward = 0.0
            episode_terminated = False
            
            for t in range(max_steps):
                # Greedy action selection (epsilon=0)
                actions = {}
                for agent_id in range(self.n_agents):
                    actions[agent_id] = self.agents[agent_id].select_action(obs[agent_id], epsilon=0.0)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                
                if terminated:
                    episode_terminated = True
                
                obs = next_obs
                episode_reward += reward
                
                if done:
                    break
            
            eval_successes.append(1 if episode_terminated else 0)
            eval_lengths.append(t + 1)
            eval_returns.append(episode_reward)
        
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
        log_dir: Optional[str] = "runs/qmix",
        eval_freq: int = 100,
        eval_episodes: int = 20,
        env_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Main training loop for QMIX.
        
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
            Directory for TensorBoard logs (default: "runs/qmix")
        eval_freq : int
            Run evaluation every N episodes (default: 100)
        eval_episodes : int
            Number of episodes to run during evaluation (default: 20)
        env_seed : Optional[int]
            Seed for environment resets (default: None)
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing training statistics
        """
        # Initialize TensorBoard writer
        writer = None
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)
        
        # Episode statistics
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        episode_losses = []
        
        # Moving averages
        window_size = 100
        success_window = []
        length_window = []
        return_window = []
        
        for episode in range(max_episodes):
            episode_seed = None if env_seed is None else env_seed + episode
            obs, info = env.reset(seed=episode_seed)
            state = self._get_global_state(env)
            
            episode_reward = 0.0
            episode_terminated = False
            episode_loss_sum = 0.0
            episode_loss_count = 0
            
            for t in range(max_steps):
                # Select actions (decentralized)
                actions = self.select_actions(obs)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                next_state = self._get_global_state(env)
                
                if terminated:
                    episode_terminated = True
                
                # Store transition
                self.store_transition(
                    state=state,
                    obs=obs,
                    actions=actions,
                    reward=reward,
                    next_state=next_state,
                    next_obs=next_obs,
                    done=done
                )
                
                # Train if buffer is large enough
                if self.total_steps % train_freq == 0:
                    if len(self.replay_memory) >= min_buffer_size:
                        loss = self.train_step()
                        if loss is not None:
                            episode_loss_sum += loss
                            episode_loss_count += 1
                
                # Update target networks periodically
                if self.total_steps > 0 and self.total_steps % self.target_update_freq == 0:
                    self.update_target_networks()
                
                # Update state
                obs = next_obs
                state = next_state
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
            
            # Update moving averages
            success_window.append(episode_success)
            length_window.append(episode_length)
            return_window.append(episode_reward)
            
            if len(success_window) > window_size:
                success_window.pop(0)
                length_window.pop(0)
                return_window.pop(0)
            
            # Log to TensorBoard
            if writer is not None:
                writer.add_scalar('episode/success', episode_success, episode)
                writer.add_scalar('episode/length', episode_length, episode)
                writer.add_scalar('episode/return', episode_reward, episode)
                
                if len(success_window) >= window_size:
                    success_rate = np.mean(success_window)
                    avg_episode_length = np.mean(length_window)
                    avg_return = np.mean(return_window)
                    
                    writer.add_scalar('metrics/success_rate', success_rate, episode)
                    writer.add_scalar('metrics/episode_length', avg_episode_length, episode)
                    writer.add_scalar('metrics/return', avg_return, episode)
                
                if episode_losses[-1] is not None:
                    writer.add_scalar('loss/training', episode_losses[-1], episode)
                
                writer.add_scalar('exploration/epsilon', self.get_epsilon(), episode)
            
            # Run evaluation
            if eval_freq > 0 and (episode + 1) % eval_freq == 0:
                eval_metrics = self.evaluate(env, n_episodes=eval_episodes, max_steps=max_steps)
                
                if writer is not None:
                    writer.add_scalar('eval/success_rate', eval_metrics['success_rate'], episode)
                    writer.add_scalar('eval/episode_length', eval_metrics['avg_episode_length'], episode)
                    writer.add_scalar('eval/return', eval_metrics['avg_return'], episode)
                
                if verbose:
                    print(f"\n[Evaluation at Episode {episode + 1}] "
                          f"Success Rate: {eval_metrics['success_rate']:.2%} | "
                          f"Avg Length: {eval_metrics['avg_episode_length']:.1f} | "
                          f"Avg Return: {eval_metrics['avg_return']:.2f}\n")
            
            # Print progress
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                success_rate = np.mean(episode_successes[-100:])
                current_epsilon = self.get_epsilon()
                print(f"Episode {episode + 1}/{max_episodes} | "
                      f"Avg Reward (last 100): {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.1f} | "
                      f"Success Rate: {success_rate:.2%} | "
                      f"Epsilon: {current_epsilon:.3f} | "
                      f"Total Steps: {self.total_steps}")
        
        # Close TensorBoard writer
        if writer is not None:
            writer.close()
            if verbose:
                print(f"\nTensorBoard logs saved to: {log_dir}")
                print(f"View with: tensorboard --logdir {log_dir}")
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_successes': episode_successes,
            'episode_losses': episode_losses,
            'total_steps': self.total_steps,
        }

