import numpy as np
import torch
from torch import nn

from config import seed, device
from network import DQNNetwork
from replay_memory import ReplayMemory
from manhattan_weight_controller import ManhattanWeightController


class DQNAgent:
    """
    Vanilla DQN with:
    - No target network
    """

    def __init__(
        self,
        env,                                                      # Gym environment (CartPole)
        epsilon_max,                                              # Start with more exploration
        epsilon_min,                                              # Minimum exploration threshold
        epsilon_decay,                                            # How fast exploration decreases
        learning_rate,                                            # Optimizer step size
        discount,                                                 # future reward discount factor
        memory_capacity,                                          # Replay buffer size
        weight_datafile_path,
        # target_update_freq                                   # optional: how often to copy weights to target (in steps)
    ):

        # Logging fields
        self.loss_history = []
        self.running_loss = 0
        self.learned_counts = 0

        # Hyperparameters
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount
        # Keep the original epsilon_max for schedule calculation
        self.epsilon_max = epsilon_max

        # Environment
        self.action_space = env.action_space                              # Saves how many actions the agent can take
        self.action_space.seed(seed)                                      # Makes random actions reproducible
        self.observation_space = env.observation_space                    # Saves the full observation space object

        # Replay buffer
        self.replay_memory = ReplayMemory(capacity=memory_capacity)

        # Q-Network
        input_dim = self.observation_space.shape[0]                       # network input = state size (4)
        output_dim = self.action_space.n                                  # network output = number of actions (2)
        self.q_network = DQNNetwork(output_dim, input_dim).to(device)


        self.criterion = torch.nn.SmoothL1Loss()

        # Manhattan-style discrete weight controller
        self.weight_controller = ManhattanWeightController(
            self.q_network,
            weight_datafile_path,
        )

        # Target Q-network (for stable targets)
        # self.target_update_freq = target_update_freq
        # self.target_network = DQNNetwork(output_dim, input_dim).to(device)
        # self.update_target_network(hard=True)

    # Action Selection (epsilon-greedy)
    def select_action(self, state):
        # exploration
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()                    # randomly picks left or right in CartPole

        # exploitation
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32, device=device)  # Convert state to tensor

        state = state.unsqueeze(0)

        with torch.no_grad():                                   # Disable gradient tracking (faster + no memory waste)
            q_values = self.q_network(state)                    # Compute Q-values: [Q_left, Q_right]
            return torch.argmax(q_values).item()                # Pick action with the highest expected reward

    # Learning step
    def learn(self, batch_size, episode_done):
        if len(self.replay_memory) < batch_size:                # Not enough samples in replay => Skip learning
            return None

        # Pulls a random batch from replay memory for training
        states, actions, next_states, rewards, dones = self.replay_memory.sample(batch_size)

        # Shape Fixing: Convert from shape (B,) [0, 1, 1, 0] → (B,1) [[0], [1], [1], [0]]
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        # self.q_network(states) → outputs all Q-values
        # .gather(1, actions) → picks only Q-values of the taken actions
        predicted_q = self.q_network(states).gather(1, actions)       # This is the Q(s, a) value from Bellman equation

        # Target Network
        # with torch.no_grad():
        #     # Use target network (stable targets) instead of online network
        #     next_q = self.target_network(next_states).max(dim=1, keepdim=True).values   # Choose max Q-value for each next state
        #     next_q[dones] = 0.0

        # Online Network (no target network)
        with torch.no_grad():
            next_q = self.q_network(next_states).max(dim=1, keepdim=True).values   # Choose max Q-value for each next state
            next_q[dones] = 0.0

        # Now build the Bellman Target
        targets = rewards + self.discount * next_q
        # compare current guess vs target (criterion is MSELoss)
        loss = self.criterion(predicted_q, targets)
        # Accumulate loss
        self.running_loss += loss.item()
        self.learned_counts += 1
        # Backprop
        self.q_network.zero_grad()
        loss.backward()                     # Compute gradients
        self.weight_controller.step()

        # If episode finished → return average loss
        if episode_done:
            avg_loss = (
                self.running_loss / self.learned_counts
                if self.learned_counts > 0 else 0.0
            )
            # Reset counters here
            self.running_loss = 0
            self.learned_counts = 0
            return avg_loss

        return None

    # Epsilon update using ε(t) = ε_min + (ε − ε_min) * exp(−λ * t)
    def update_epsilon(self, steps_done):
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay * steps_done)

    # Model saving
    def save(self, path):
        torch.save(self.q_network.state_dict(), path)             # Stores parameters (weights) to a file

    # Target network management
    # def update_target_network(self, hard=True, tau=1.0):
    #     """Update target network parameters.
    #
    #     If hard=True (default), copy online network parameters to target.
    #     If hard=False, perform soft update: target = tau * online + (1-tau) * target
    #     """
    #     if hard or tau == 1.0:
    #         self.target_network.load_state_dict(self.q_network.state_dict())
    #     else:
    #         # Soft update
    #         for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
    #             target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    #
    #     # Put target network in eval mode (no dropout/batchnorm training behavior)
    #     self.target_network.eval()
