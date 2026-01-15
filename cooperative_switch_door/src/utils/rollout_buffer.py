import torch


class RolloutBuffer:
    """
    The rollout buffer is just a memory box where we store what happened at each step so PPO can learn later.
    """
    def __init__(self, obs_dim, state_dim, n_agents, buffer_size):
        """
        Initialize the rollout buffer.
        """
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.buffer_size = buffer_size

        self.obs = None         # observations per agent per step
        self.states = None      # global states per step
        self.actions = None     # actions taken by each agent
        self.log_probs = None   # log probabilities of actions taken
        self.values = None      # centralized critic values
        self.rewards = None     # rewards per step
        self.dones = None       # done flags per step

        self.clear()

    def clear(self):
        """
        Clear the buffer.
        """
        self.obs = []
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, obs, state, actions, log_probs, value, reward, done):
        """
        Add a new experience to the buffer.
        """
        self.obs.append([obs[i] for i in range(self.n_agents)])
        self.actions.append([actions[i] for i in range(self.n_agents)])
        self.log_probs.append([log_probs[i] for i in range(self.n_agents)])
        self.states.append(state)
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def get(self):
        """
        
        :return:
        """
        return {
            "obs": torch.tensor(self.obs, dtype=torch.float32),
            "states": torch.tensor(self.states, dtype=torch.float32),
            "actions": torch.tensor(self.actions, dtype=torch.long),
            "log_probs": torch.tensor(self.log_probs, dtype=torch.float32),
            "values": torch.tensor(self.values, dtype=torch.float32),
            "rewards": torch.tensor(self.rewards, dtype=torch.float32),
            "dones": torch.tensor(self.dones, dtype=torch.float32),
        }

