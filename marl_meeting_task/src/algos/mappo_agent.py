import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from marl_meeting_task.src.models.actor_critic import ActorNetwork, CriticNetwork, CentralizedCritic
from marl_meeting_task.src.config import device


class MAPPOAgent:
    """
    MAPPOAgent wraps actor(s) and critic(s). Supports shared actor and optional
    centralized critic. Designed to be lightweight and consistent with other agents.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        share_actor: bool = True,
        use_centralized_critic: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.share_actor = share_actor
        self.use_centralized_critic = use_centralized_critic
        self.device = device if device is not None else device

        # Actors
        if self.share_actor:
            self.actor = ActorNetwork(obs_dim, action_dim, hidden_sizes=(hidden_dim, hidden_dim)).to(device)
        else:
            self.actors = nn.ModuleList([
                ActorNetwork(obs_dim, action_dim, hidden_sizes=(hidden_dim, hidden_dim)).to(device)
                for _ in range(n_agents)
            ])

        # Critics
        if self.use_centralized_critic:
            # joint observation dimension: concatenate per-agent observations
            joint_dim = obs_dim * n_agents
            self.critic = CentralizedCritic(joint_dim, n_agents, hidden_sizes=(hidden_dim * 2, hidden_dim * 2)).to(device)
        else:
            # per-agent critic
            if self.share_actor:
                # single critic per agent (not shared) - create ModuleList for explicitness
                self.critics = nn.ModuleList([
                    CriticNetwork(obs_dim, hidden_sizes=(hidden_dim, hidden_dim)).to(device)
                    for _ in range(n_agents)
                ])
            else:
                self.critics = nn.ModuleList([
                    CriticNetwork(obs_dim, hidden_sizes=(hidden_dim, hidden_dim)).to(device)
                    for _ in range(n_agents)
                ])

    def _prep_obs_tensor(self, obs: Dict[int, np.ndarray]) -> torch.Tensor:
        """Convert dict of per-agent observations to tensor of shape [n_agents, obs_dim]."""
        obs_list = [obs[i] for i in range(self.n_agents)]
        obs_arr = np.stack(obs_list, axis=0).astype(np.float32)
        return torch.tensor(obs_arr, dtype=torch.float32, device=device)

    def select_action(self, obs: Dict[int, np.ndarray], deterministic: bool = False) -> Tuple[Dict[int, int], Dict[int, float], Dict[int, float]]:
        """
        Select actions for all agents given a single timestep observation dict.
        Returns (actions, log_probs, values) as dicts keyed by agent id.
        """
        assert isinstance(obs, dict) and len(obs) == self.n_agents, "obs must be a dict with one entry per agent"

        obs_tensor = self._prep_obs_tensor(obs)  # [n_agents, obs_dim]

        actions = {}
        log_probs = {}
        values = {}

        # Actor forward
        if self.share_actor:
            logits = self.actor(obs_tensor)  # [n_agents, action_dim]
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                acts = torch.argmax(logits, dim=1)
            else:
                acts = dist.sample()
            lps = dist.log_prob(acts)
        else:
            acts_list = []
            lps_list = []
            for i in range(self.n_agents):
                logits = self.actors[i](obs_tensor[i:i+1])  # [1, action_dim]
                dist = torch.distributions.Categorical(logits=logits)
                if deterministic:
                    a = torch.argmax(logits, dim=1)
                else:
                    a = dist.sample()
                acts_list.append(int(a.item()))
                lps_list.append(float(dist.log_prob(a).item()))
            # convert to tensors for values computation
            acts = torch.tensor(acts_list, device=device, dtype=torch.long)
            lps = torch.tensor(lps_list, device=device, dtype=torch.float32)

        # Critic forward
        if self.use_centralized_critic:
            joint_obs = obs_tensor.view(1, -1)  # [1, joint_dim]
            vals = self.critic(joint_obs).squeeze(0)  # [n_agents]
        else:
            vals_list = []
            for i in range(self.n_agents):
                v = self.critics[i](obs_tensor[i:i+1]).squeeze(0)  # scalar tensor
                vals_list.append(v)
            vals = torch.stack(vals_list, dim=0)

        # Fill dicts
        for i in range(self.n_agents):
            actions[i] = int(acts[i].item())
            log_probs[i] = float(lps[i].item())
            values[i] = float(vals[i].item())

        return actions, log_probs, values

    def evaluate_actions(self, obs_batch: torch.Tensor, actions_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate a batch of observations and actions for training.
        obs_batch: [batch, n_agents, obs_dim]
        actions_batch: [batch, n_agents]
        Returns: (log_probs, values, entropies) flattened per agent: shape [batch * n_agents]
        """
        batch_size = obs_batch.shape[0]
        n_agents = self.n_agents
        obs_batch = obs_batch.to(device)
        actions_batch = actions_batch.to(device)

        # Flatten agents into batch dimension for actor if shared
        if self.share_actor:
            flat_obs = obs_batch.view(batch_size * n_agents, self.obs_dim)
            logits = self.actor(flat_obs)  # [batch*n_agents, action_dim]
            dist = torch.distributions.Categorical(logits=logits)
            flat_actions = actions_batch.view(-1)
            log_probs = dist.log_prob(flat_actions)
            ent = dist.entropy()
        else:
            # per-agent actors
            log_probs_list = []
            ent_list = []
            for i in range(n_agents):
                obs_i = obs_batch[:, i, :]
                logits = self.actors[i](obs_i)
                dist = torch.distributions.Categorical(logits=logits)
                acts = actions_batch[:, i]
                log_probs_list.append(dist.log_prob(acts))
                ent_list.append(dist.entropy())
            log_probs = torch.stack(log_probs_list, dim=1).view(-1)
            ent = torch.stack(ent_list, dim=1).view(-1)

        # Values
        if self.use_centralized_critic:
            # joint obs: [batch, n_agents*obs_dim]
            joint = obs_batch.view(batch_size, -1)
            vals = self.critic(joint)  # [batch, n_agents]
            flat_vals = vals.view(-1)
        else:
            vals_list = []
            for i in range(n_agents):
                v = self.critics[i](obs_batch[:, i, :])  # [batch]
                vals_list.append(v)
            vals = torch.stack(vals_list, dim=1)  # [batch, n_agents]
            flat_vals = vals.view(-1)

        return log_probs, flat_vals, ent

    def get_parameters(self):
        """Return parameter iterables for actor and critic optimizers."""
        params = {}
        if self.share_actor:
            params['actor'] = self.actor.parameters()
        else:
            params['actor'] = self.actors.parameters()

        if self.use_centralized_critic:
            params['critic'] = self.critic.parameters()
        else:
            params['critic'] = self.critics.parameters()

        return params

    def save(self, path: str):
        """Save actor and critic state dicts."""
        to_save = {}
        if self.share_actor:
            to_save['actor'] = self.actor.state_dict()
        else:
            to_save['actors'] = self.actors.state_dict()

        if self.use_centralized_critic:
            to_save['critic'] = self.critic.state_dict()
        else:
            to_save['critics'] = self.critics.state_dict()

        torch.save(to_save, path)

    def load(self, path: str, map_location=None):
        data = torch.load(path, map_location=map_location)
        if self.share_actor and 'actor' in data:
            self.actor.load_state_dict(data['actor'])
        elif not self.share_actor and 'actors' in data:
            self.actors.load_state_dict(data['actors'])

        if self.use_centralized_critic and 'critic' in data:
            self.critic.load_state_dict(data['critic'])
        elif not self.use_centralized_critic and 'critics' in data:
            self.critics.load_state_dict(data['critics'])

