import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym


class MyCartPoleEnv:
    def __init__(self, render_mode=None, seed=None, length=None):
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.seed = seed

        # Set seeds for reproducibility
        if seed is not None:
            self.env.reset(seed=seed)
            self.env.action_space.seed(seed)

        #  Modify pole length if provided
        if length is not None:
            base_env = self.env.unwrapped
            base_env.length = length
            base_env.polemass_length = base_env.masspole * length  # VERY IMPORTANT

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