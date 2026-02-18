import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import config
from envs.agent import DQNAgent
from envs.cartepole import CartPoleEnv


class Trainer:
    def __init__(self, hyperparams):
        # Load parameters
        self.discount_factor = hyperparams["discount_factor"]           # Bellman γ (future reward weight)
        self.batch_size = hyperparams["batch_size"]                     # Number of experiences per learning step
        self.max_episodes = hyperparams["max_episodes"]                 # number of episode for training or testing
        self.max_steps = hyperparams["max_steps"]                       # Episode timeout
        self.epsilon_max = hyperparams["epsilon_max"]                   # Initial exploration rate
        self.epsilon_min = hyperparams["epsilon_min"]                   # Minimum allowed epsilon
        self.epsilon_decay = hyperparams["epsilon_decay"]               # Exploration decay speed
        self.memory_capacity = hyperparams["memory_capacity"]           # Replay buffer size

        self.env = CartPoleEnv(render_mode=None)
        self.agent = DQNAgent(
            env=self.env,
            epsilon_max=self.epsilon_max,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            discount=self.discount_factor,
            memory_capacity=self.memory_capacity,
        )


    def train(self):
        total_steps = 0                                     # Count steps across all episodes (used for epsilon decay

        for episode in range(1, self.max_episodes + 1):
            state = self.env.reset()        # Initial observation from environment
            done = False                                    # Episode ended because of failure
            episode_reward = 0                              # Episode ended because of failure
            step_counter = 0                                # Step counter inside episode
            while not done:                                 # The agent keeps taking steps until episode ends
                action = self.agent.select_action(state)    # Using epsilon-greedy strategy—exploration or exploitation
                next_state, reward, done = self.env.step(action) # Environment responds
                self.agent.replay_memory.store(state, action, next_state, reward, done) # This is essential for off-policy learning
                if len(self.agent.replay_memory) > self.batch_size:         # Only learn when enough samples collected
                    self.agent.learn(self.batch_size)
                # Tracking step and reward progress
                state = next_state
                episode_reward += reward
                step_counter += 1

            total_steps += step_counter
            # Update epsilon (step-based)
            self.agent.update_epsilon(total_steps)

            # Shows training progress in readable way
            print(
                f"Episode: {episode}, "
                f"Steps: {step_counter}, "
                f"Reward: {episode_reward:.2f}, "
                f"Epsilon: {self.agent.epsilon:.2f}"
            )