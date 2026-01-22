import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    """
    Small MLP actor producing logits for a discrete action space.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(64, 64)):
        super(ActorNetwork, self).__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return action logits of shape [batch, action_dim]."""
        return self.network(obs)


class CriticNetwork(nn.Module):
    """
    Per-agent value network that outputs a single scalar value.
    """

    def __init__(self, obs_dim: int, hidden_sizes=(64, 64)):
        super(CriticNetwork, self).__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return value estimates of shape [batch]."""
        return self.network(obs).squeeze(-1)


class CentralizedCritic(nn.Module):
    """
    Centralized critic that consumes joint observations and returns one value per agent.
    For simplicity the output dimension is `n_agents` (a separate value per agent).
    """

    def __init__(self, joint_obs_dim: int, n_agents: int, hidden_sizes=(128, 128)):
        super(CentralizedCritic, self).__init__()
        self.n_agents = n_agents
        layers = []
        last = joint_obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, n_agents))
        self.network = nn.Sequential(*layers)

    def forward(self, joint_obs: torch.Tensor) -> torch.Tensor:
        """Return values of shape [batch, n_agents]."""
        return self.network(joint_obs)
