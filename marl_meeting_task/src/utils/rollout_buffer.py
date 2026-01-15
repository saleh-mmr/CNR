import numpy as np
import torch


class RolloutBuffer:
    """
    Simple on-policy rollout buffer for MAPPO. Stores per-step, per-agent data and
    provides advantage/return computation (GAE) and minibatch iteration.
    """

    def __init__(self, rollout_length: int, n_agents: int, obs_dim: int, device: torch.device):
        self.rollout_length = rollout_length
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.device = device

        self.reset()

    def reset(self):
        self.observations = []  # list of dicts per timestep
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs: dict, actions: dict, log_probs: dict, rewards: float, dones: bool, values: dict):
        """Add a single timestep (per-agent dicts) to the buffer."""
        # obs: dict[int -> np.array], actions/log_probs/values: dicts keyed by agent
        self.observations.append(obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.values.append(values)

    def compute_returns_and_advantages(self, last_values: dict, gamma: float, gae_lambda: float, use_gae: bool = True):
        """
        Compute returns and advantages and flatten data to arrays shaped [T * n_agents, ...].
        last_values: dict of last value per agent (from critic)
        """
        T = len(self.rewards)
        n = self.n_agents

        # convert values per timestep per agent to numpy arrays
        vals = np.zeros((T, n), dtype=np.float32)
        for t in range(T):
            for i in range(n):
                vals[t, i] = float(self.values[t][i])
        last_vals_arr = np.array([float(last_values[i]) for i in range(n)], dtype=np.float32)

        returns = np.zeros((T, n), dtype=np.float32)
        advantages = np.zeros((T, n), dtype=np.float32)

        next_values = last_vals_arr
        next_adv = np.zeros(n, dtype=np.float32)

        for t in reversed(range(T)):
            mask = 0.0 if self.dones[t] else 1.0
            rewards = float(self.rewards[t])
            delta = rewards + gamma * next_values * mask - vals[t]
            if use_gae:
                next_adv = delta + gamma * gae_lambda * mask * next_adv
                advantages[t] = next_adv
                returns[t] = advantages[t] + vals[t]
            else:
                returns[t] = rewards + gamma * next_values * mask
                advantages[t] = returns[t] - vals[t]
            next_values = vals[t]

        # Flatten: time-major then agents
        flat_obs = []
        flat_actions = []
        flat_log_probs = []
        flat_returns = []
        flat_advantages = []

        for t in range(T):
            for i in range(n):
                flat_obs.append(self.observations[t][i].astype(np.float32))
                flat_actions.append(self.actions[t][i])
                flat_log_probs.append(self.log_probs[t][i])
                flat_returns.append(returns[t, i])
                flat_advantages.append(advantages[t, i])

        # Convert to numpy arrays
        self.flat_obs = np.stack(flat_obs, axis=0)  # [T*n, obs_dim]
        self.flat_actions = np.array(flat_actions, dtype=np.int64)  # [T*n]
        self.flat_log_probs = np.array(flat_log_probs, dtype=np.float32)
        self.flat_returns = np.array(flat_returns, dtype=np.float32)
        self.flat_advantages = np.array(flat_advantages, dtype=np.float32)

    def get_batches(self, mini_batch_size: int, shuffle: bool = True):
        """Yield minibatches (as numpy arrays) for PPO updates."""
        N = self.flat_obs.shape[0]
        indices = np.arange(N)
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, N, mini_batch_size):
            end = start + mini_batch_size
            mb_idx = indices[start:end]
            yield (
                torch.tensor(self.flat_obs[mb_idx], dtype=torch.float32, device=self.device),
                torch.tensor(self.flat_actions[mb_idx], dtype=torch.long, device=self.device),
                torch.tensor(self.flat_log_probs[mb_idx], dtype=torch.float32, device=self.device),
                torch.tensor(self.flat_returns[mb_idx], dtype=torch.float32, device=self.device),
                torch.tensor(self.flat_advantages[mb_idx], dtype=torch.float32, device=self.device),
            )

