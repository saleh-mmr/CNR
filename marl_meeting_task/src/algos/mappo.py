import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any

from marl_meeting_task.src.algos.mappo_agent import MAPPOAgent
from marl_meeting_task.src.utils.rollout_buffer import RolloutBuffer
from marl_meeting_task.src.config import device
from marl_meeting_task.src.utils.logger import Logger


class MAPPO:
    """
    Minimal MAPPO trainer consistent with repository conventions.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        config: Dict[str, Any],
        logger: Optional[Logger] = None,
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device
        self.logger = logger or Logger()

        # Create agent (shared actor by default)
        self.agent = MAPPOAgent(
            n_agents=n_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=config.get('hidden_dim', 64),
            share_actor=config.get('share_actor', True),
            use_centralized_critic=config.get('use_centralized_critic', True),
            device=self.device,
        )

        # Optimizers
        lr_actor = config.get('lr_actor', 3e-4)
        lr_critic = config.get('lr_critic', 3e-4)
        self.actor_optimizer = optim.Adam(self.agent.get_parameters()['actor'], lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.agent.get_parameters()['critic'], lr=lr_critic)

        # Rollout buffer
        self.rollout_length = config.get('rollout_length', 64)
        self.buffer = RolloutBuffer(self.rollout_length, n_agents, obs_dim, device=self.device)

        # PPO hyperparams
        self.clip_eps = config.get('clip_eps', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.mini_batch_size = config.get('mini_batch_size', 64)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)

    def collect_rollouts(self, env, num_steps: Optional[int] = None):
        """Collect a rollout of length `self.rollout_length` from `env`."""
        env_obs, _ = env.reset()
        for t in range(self.rollout_length):
            actions, log_probs, values = self.agent.select_action(env_obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            # Store into buffer
            self.buffer.add(env_obs, actions, log_probs, float(reward), done, values)
            env_obs = next_obs
            if done:
                env_obs, _ = env.reset()

        # Get last values for bootstrap
        _, _, last_values = self.agent.select_action(env_obs, deterministic=True)
        return last_values

    def update(self, last_values: Dict[int, float]):
        """Run PPO update using data in the buffer."""
        # Compute returns and advantages
        self.buffer.compute_returns_and_advantages(last_values, self.gamma, self.gae_lambda, use_gae=True)

        # Normalize advantages
        adv = self.buffer.flat_advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.buffer.flat_advantages = adv

        losses = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}

        N = self.buffer.flat_obs.shape[0]
        for epoch in range(self.ppo_epochs):
            for mb in self.buffer.get_batches(self.mini_batch_size, shuffle=True):
                obs_mb, actions_mb, old_log_probs_mb, returns_mb, adv_mb = mb

                # Reshape obs_mb to [batch, n_agents, obs_dim]
                batch_size = obs_mb.shape[0] // self.n_agents
                obs_mb_agents = obs_mb.view(batch_size, self.n_agents, self.obs_dim)
                actions_mb_agents = actions_mb.view(batch_size, self.n_agents)

                # Evaluate current policy
                new_log_probs, values, entropy = self.agent.evaluate_actions(obs_mb_agents, actions_mb_agents)

                # PPO ratio
                ratio = torch.exp(new_log_probs - old_log_probs_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(values, returns_mb)
                entropy_loss = -entropy.mean()

                # Optimize actor (include entropy regularization in actor loss)
                actor_params = list(self.agent.get_parameters()['actor'])
                actor_loss = policy_loss + self.entropy_coef * entropy_loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_params, max_norm=self.max_grad_norm)
                self.actor_optimizer.step()

                # Optimize critic (value loss only)
                critic_params = list(self.agent.get_parameters()['critic'])
                self.critic_optimizer.zero_grad()
                (self.value_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(critic_params, max_norm=self.max_grad_norm)
                self.critic_optimizer.step()

                losses['policy_loss'] += float(policy_loss.item())
                losses['value_loss'] += float(value_loss.item())
                losses['entropy'] += float(entropy.mean().item())

        # Average losses
        num_updates = self.ppo_epochs * (N // self.mini_batch_size + int(N % self.mini_batch_size > 0))
        for k in losses:
            losses[k] /= max(1, num_updates)

        # Clear buffer
        self.buffer.reset()
        return losses

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.agent.save(path)

    def load(self, path: str):
        self.agent.load(path)

