import torch
import torch.nn as nn
import torch.nn.functional as F


class MixingNetwork(nn.Module):
    """
    QMIX Mixing Network (single-layer monotonic variant).

    This simplified mixing network uses a single state-conditioned layer to
    combine individual agent Q-values into a joint Q-value Q_tot. The mixing
    weights are produced by a hypernetwork (`hyper_w1`) and passed through
    `softplus` to ensure they are strictly non-negative, which preserves the
    monotonicity constraint: increasing any agent Q_i cannot decrease Q_tot.

    Architecture (this file's variant):
    - hyper_w1: state -> [n_agents] (per-agent weights)
    - hyper_b1: state -> [1] (global bias)
    - Q_tot = sum_i softplus(w_i(state)) * Q_i + b(state)
    """
    
    def __init__(
        self,
        n_agents: int = 2,
        state_dim: int = 6,  # Global state: [a1_x, a1_y, a2_x, a2_y, g_x, g_y]
        mixing_hidden_dim: int = 64,
    ):
        """
        Initialize single-layer mixing network.
        """
        super(MixingNetwork, self).__init__()
        
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_hidden_dim = mixing_hidden_dim
        
        # Hypernetwork producing per-agent weights. We keep the name
        # `hyper_w1` but change the final output size to `n_agents` so that
        # for each state we get one weight per agent.
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, n_agents),
        )
        
        # Hypernetwork producing a single scalar bias per state (global bias).
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, mixing_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixing_hidden_dim, 1),
        )
        
        # Note: removed hyper_w2 and hyper_b2 (two-layer mixing) as requested.

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the single-layer mixing network.

        Parameters:
        -----------
        agent_qs : torch.Tensor
            Individual agent Q-values of shape [batch_size, n_agents]
        states : torch.Tensor
            Global states of shape [batch_size, state_dim]
            
        Returns:
        --------
        torch.Tensor
            Joint Q-value Q_tot of shape [batch_size, 1]
        """
        # Expect agent_qs: [batch, n_agents]
        batch_size = agent_qs.shape[0]
        assert agent_qs.shape[1] == self.n_agents, \
            f"Expected agent_qs second dim to be n_agents={self.n_agents}, got {agent_qs.shape[1]}"

        # Produce positive weights per agent from the state
        w = F.softplus(self.hyper_w1(states))  # [batch_size, n_agents]
        # Produce scalar bias per state
        b = self.hyper_b1(states).view(batch_size, 1)  # [batch_size, 1]

        # Compute weighted sum: ensure broadcasting shapes line up
        # agent_qs: [batch, n_agents], w: [batch, n_agents]
        q_tot = (agent_qs * w).sum(dim=1, keepdim=True) + b  # [batch, 1]
        return q_tot
