import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from envs.agent_simple_test import DQNAgent
from envs.cartepole import CartPoleEnv

agent = DQNAgent(
    n_action_space = 2,
    n_observation_space = 4,
    epsilon_max = 1.0,  # Start with more exploration
    epsilon_min = 0.01,  # Minimum exploration threshold
    epsilon_decay = 0.000005, # How fast exploration decreases
    discount = 0.99,  # future reward discount factor
    memory_capacity = 100000  # Replay buffer size
)

env = CartPoleEnv()
total_reward = 0
total_steps = 0
for episode in range(5000):
    state = env.reset()
    done = False
    episode_reward = 0
    episode_steps = 0
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        state = next_state
        episode_reward += reward
        episode_steps += 1
        total_steps += 1
        if len(agent.cart_pole_memory) > 30:
            agent.learn(30, total_steps)

    print(
        f"Episode: {episode}, "
        f"Steps: {total_steps}, "
        f"Reward CP: {episode_reward:.2f}, "
    )
