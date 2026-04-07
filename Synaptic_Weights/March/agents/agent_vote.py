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
        self.criterion = nn.MSELoss(reduction='none')

        self.weight_controller = SynapticWeightController(self.q_network)

    def _assign_majority_sign_grads(self, sign_sums, params_dict):
        for name, param in params_dict.items():
            vote = torch.where(
                sign_sums[name] > 0,
                torch.ones_like(sign_sums[name]),
                -torch.ones_like(sign_sums[name]),
            )
            param.grad = vote.to(dtype=param.dtype)

    def _per_sample_majority_vote_vmap(self, states, actions, targets, params_dict, buffers_dict):
        from torch.func import functional_call, grad, vmap

        def single_loss(params, buffers, state, action, target):
            q_values = functional_call(self.q_network, (params, buffers), (state.unsqueeze(0),))
            pred_q = q_values.gather(1, action.view(1, 1))
            return self.criterion(pred_q, target.view(1, 1)).mean()

        grad_fn = grad(single_loss)
        squeezed_actions = actions.squeeze(1)
        squeezed_targets = targets.squeeze(1)
        per_sample_grads = vmap(grad_fn, in_dims=(None, None, 0, 0, 0))(
            params_dict,
            buffers_dict,
            states,
            squeezed_actions,
            squeezed_targets,
        )

        sign_sums = {}
        for name, grad_tensor in per_sample_grads.items():
            clean_grad = torch.nan_to_num(grad_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
            sign_sums[name] = torch.sign(clean_grad).sum(dim=0)

        self._assign_majority_sign_grads(sign_sums, params_dict)

    def _per_sample_majority_vote_fallback(self, states, actions, targets, params_dict):
        sign_sums = {
            name: torch.zeros_like(param, device=param.device)
            for name, param in params_dict.items()
        }
        params_list = list(params_dict.values())
        names_list = list(params_dict.keys())

        for i in range(states.shape[0]):
            q_values = self.q_network(states[i:i + 1])
            pred_q = q_values.gather(1, actions[i:i + 1])
            sample_loss = self.criterion(pred_q, targets[i:i + 1]).mean()
            sample_grads = torch.autograd.grad(sample_loss, params_list, allow_unused=True)

            for name, grad_tensor in zip(names_list, sample_grads):
                if grad_tensor is None:
                    continue
                clean_grad = torch.nan_to_num(grad_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
                sign_sums[name] += torch.sign(clean_grad)

        self._assign_majority_sign_grads(sign_sums, params_dict)

    def _apply_majority_vote_grads(self, states, actions, targets):
        params_dict = {
            name: param
            for name, param in self.q_network.named_parameters()
            if param.requires_grad
        }
        buffers_dict = {
            name: buf
            for name, buf in self.q_network.named_buffers()
        }

        self.q_network.zero_grad(set_to_none=True)

        try:
            self._per_sample_majority_vote_vmap(states, actions, targets, params_dict, buffers_dict)
        except Exception:
            self._per_sample_majority_vote_fallback(states, actions, targets, params_dict)

    # Action Selection (epsilon-greedy)
    def select_action(self, state, ap_index, epsilon=None):

        if epsilon is None:
            epsilon = self.epsilon

        # exploration
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.action_space_dim)

        # exploration
        state = torch.as_tensor(state, dtype=torch.float32, device=config.device).unsqueeze(0)
        with torch.no_grad():
            # weight_matrix = self.q_network.state_dict()["FC.0.weight"]
            # print(weight_matrix[0, 0].item())
            self.weight_controller.load_weights(ap_index)
            q_values = self.q_network(state)
            # print(f"Q-values for AP index {ap_index}: {q_values.cpu().numpy()}")
        return torch.argmax(q_values, dim=1).item()        # exploration



    def select_action_test(self, state):
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
        actions = actions.long().unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.bool().unsqueeze(1)

        self.weight_controller.load_weights(ap_index)  # Load current weights from the controller before forward pass

        # Max future reward if the episode is not terminal
        with torch.no_grad():
            next_q = self.q_network(next_states).max(dim=1, keepdim=True).values   # Choose max Q-value for each next state
            next_q[dones] = 0.0
        targets = rewards + self.discount * next_q

        # Compute per-sample grads, vote by sign over the batch, and write +/-1 into param.grad.
        self._apply_majority_vote_grads(states, actions, targets)

        self.weight_controller.step(ap_index)
        return None

    # Epsilon update using ε(t) = ε_min + (ε_max − ε_min) * exp(−λ * t)
    def update_epsilon(self, steps_done):
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.epsilon_decay * steps_done)

    # Model saving
    def save(self, path):
        torch.save(self.q_network.state_dict(), path)             # Stores parameters (weights) to a file