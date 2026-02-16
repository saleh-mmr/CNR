import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
from utils import config

class MountainCarEnv:
    """
    Wrapper around Gymnasium MountainCar-v0.
    Compatible with DQN (discrete action space).
    """

    def __init__(self, render_mode=None):
        self.env = gym.make("MountainCar-v0", render_mode=render_mode)

        # Set seeds for reproducibility
        self.env.reset(seed=config.seed)
        self.env.action_space.seed(config.seed)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self):
        state, _ = self.env.reset()
        return state

    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()