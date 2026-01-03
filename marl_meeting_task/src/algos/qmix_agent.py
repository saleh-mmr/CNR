import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from marl_meeting_task.src.utils.qvalue_network import QValueNetwork
from marl_meeting_task.src.config import device


class QMIXAgent:
    """
    QMIX Agent Component.
    
    Each agent has its own Q-network that takes local observations
    and outputs Q-values for actions. These are used during execution
    (decentralized) and combined via the mixing network during training
    (centralized).
    """
    
    def __init__(
        self,
        agent_id: int,
        input_dim: int = 6,
        num_actions: int = 5,
        hidden_dim: int = 64,
    ):
        """
        Initialize QMIX Agent.
        
        Parameters:
        -----------
        agent_id : int
            Unique identifier for this agent
        input_dim : int
            Dimension of observation vector (default: 6)
        num_actions : int
            Number of possible actions (default: 5)
        hidden_dim : int
            Hidden layer dimension for Q-network (default: 64)
        """
        self.agent_id = agent_id
        self.num_actions = num_actions
        
        # Agent Q-network (main)
        self.q_network = QValueNetwork(
            input_dim=input_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim
        ).to(device)
        
        # Agent Q-network (target)
        self.target_network = QValueNetwork(
            input_dim=input_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim
        ).to(device)
        self.update_target_network()  # Initialize with same weights
    
    def update_target_network(self) -> None:
        """Copy weights from main Q-network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Parameters:
        -----------
        obs : np.ndarray
            Local observation vector
        epsilon : float
            Exploration probability
            
        Returns:
        --------
        int
            Selected action index
        """
        if np.random.random() < epsilon:
            # Exploration: random action
            return np.random.randint(0, self.num_actions)
        else:
            # Exploitation: greedy action
            with torch.no_grad():
                obs_tensor = torch.as_tensor(
                    obs,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)  # Add batch dimension
                q_values = self.q_network(obs_tensor)
                return q_values.argmax().item()

