import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch import nn
import torch
from utils import config
from memory.replay_memory import ReplayMemory
from network.network import DQNNetwork
from controller.synaptic_weight_controller import SynapticWeightController


class DQNAgent:
    def __init__(
        self,
        cartpole_env,                                             # Gym cartpole environment
        mountaincar_env,                                          # Gym mountaincar environment
        epsilon_max,                                              # Start with more exploration
        epsilon_min,                                              # Minimum exploration threshold
        epsilon_decay,                                            # How fast exploration decreases
        discount,                                                 # future reward discount factor
        memory_capacity,                                          # Replay buffer size
    ):
        # Hyperparameters
        self.epsilon = epsilon_max
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount

        # Environment
        self.cartpole_env = cartpole_env
        self.mountaincar_env = mountaincar_env
        assert cartpole_env.action_space.n == mountaincar_env.action_space.n, "Action space dimensions must match"
        assert cartpole_env.observation_space.shape[0] == mountaincar_env.observation_space.shape[0], "Observation space dimensions must match"
        self.action_space_dim = cartpole_env.action_space.n
        self.observation_space_dim = cartpole_env.observation_space.shape[0]


        # Replay buffer
        self.cartpole_memory = ReplayMemory(capacity=memory_capacity)
        self.mountaincar_memory = ReplayMemory(capacity=memory_capacity)
        self.replay_memory = [self.cartpole_memory, self.mountaincar_memory]

        # Q-Network
        input_dim = self.observation_space_dim                                # network input = state size (4)
        output_dim = self.action_space_dim                                    # network output = number of actions (2)
        self.q_network = DQNNetwork(output_dim, input_dim).to(config.device)

        # use a squared-error loss just to get gradients,
        self.criterion = nn.MSELoss()

        self.weight_controller = SynapticWeightController(self.q_network)

    # Action Selection (epsilon-greedy)
    def select_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon

        # exploration
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.action_space_dim)

        # exploration
        state = torch.as_tensor(state, dtype=torch.float32, device=config.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)

        return torch.argmax(q_values, dim=1).item()        # exploration

    # Learning step
    def learn(self, batch_size, ap_index):
        if len(self.replay_memory[ap_index]) < batch_size:                # Not enough future in replay => Skip learning
            return None

        # Pulls a random batch from replay memory for training
        states, actions, next_states, rewards, dones = self.replay_memory[ap_index].sample(batch_size)

        # Shape Fixing: Convert from shape (B,) [0, 1, 1, 0] → (B,1) [[0], [1], [1], [0]]
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        self.weight_controller.load_weights(ap_index)  # Load current weights from the controller before forward pass
        # self.q_network(states) → outputs all Q-values
        # .gather(1, actions) → picks only Q-values of the taken actions
        q_all = self.q_network(states)
        predicted_q = q_all.gather(1, actions)

        # Max future reward if the episode is not terminal
        with torch.no_grad():
            next_q = self.q_network(next_states).max(dim=1, keepdim=True).values   # Choose max Q-value for each next state
            next_q[dones] = 0.0
        targets = rewards + self.discount * next_q

        # compare current guess vs target (criterion is MSELoss)
        loss = self.criterion(predicted_q, targets)

        # Clear old gradients
        for param in self.q_network.parameters():
            if param.grad is not None:
                param.grad.zero_()
        loss.backward()
        self.weight_controller.step(ap_index)

        return loss.item()

    # Epsilon update using ε(t) = ε_min + (ε_max − ε_min) * exp(−λ * t)
    def update_epsilon(self, steps_done):
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay * steps_done)

    # Model saving
    def save(self, path):
        torch.save(self.q_network.state_dict(), path)             # Stores parameters (weights) to a file