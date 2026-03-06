import os
import random
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.agent import DQNAgent
from envs.cartpole import CartPoleEnv
from envs.mountaincar import MountainCarEnv


class Trainer:
    def __init__(self, hyperparams, seed):
        # Load parameters
        self.discount_factor = hyperparams["discount_factor"]  # Bellman γ (future reward weight)
        self.batch_size = hyperparams["batch_size"]  # Number of experiences per learning step
        self.max_episodes = hyperparams["max_episodes"]  # number of episode for training or testing
        self.max_steps = hyperparams["max_steps"]  # Episode timeout
        self.epsilon_max = hyperparams["epsilon_max"]  # Initial exploration rate
        self.epsilon_min = hyperparams["epsilon_min"]  # Minimum allowed epsilon
        self.epsilon_decay = hyperparams["epsilon_decay"]  # Exploration decay speed
        self.memory_capacity = hyperparams["memory_capacity"]  # Replay buffer size
        self.seed = seed
        self.cart_pole_env = MountainCarEnv(render_mode=None, seed=seed)
        self.agent = DQNAgent(
            n_action_space=self.cart_pole_env.action_space.n,
            n_observation_space=self.cart_pole_env.observation_space.shape[0],
            epsilon_max=self.epsilon_max,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            discount=self.discount_factor,
            memory_capacity=self.memory_capacity,
        )

    def train(self):
        total_steps = 0
        total_reward = []
        loss_track = []
        best_so_far = self.max_steps
        self.prefill_replay_memory(20000)

        for episode in range(1, self.max_episodes + 1):
            # Initial observation from environment
            state = self.cart_pole_env.reset()
            # Flags to track episode completion for each environment
            done = False
            # Total reward accumulated in this episode each environment (for logging)
            episode_reward = 0
            step_counter = 0  # Step counter inside episode
            while not done and step_counter < self.max_steps:
                # For each environment, if it's not done, select action, step, store experience, and accumulate reward
                action_cp = self.agent.select_action(state)
                # Step in the environment and get next state, reward, and done flag
                next_state, reward, done = self.cart_pole_env.step(action_cp)
                # Store experience in the corresponding replay memory
                self.agent.replay_memory.store(state, action_cp, next_state, reward, done)
                state = next_state
                episode_reward += reward

                loss = self.agent.learn(self.batch_size)
                loss_track.append(loss)

                total_steps += 1
                step_counter += 1
                # Update epsilon (step-based)
                self.agent.update_epsilon(total_steps)

            total_reward.append(episode_reward)
            # Shows training progress in readable way
            print(
                f"Episode: {episode}, "
                f"Steps: {step_counter}, "
                f"Reward CP: {episode_reward:.2f}, "
                f"Epsilon: {self.agent.epsilon:.2f}"
            )

            # SAVE BEST MODEL
            if abs(episode_reward) < best_so_far:
                best_so_far = abs(episode_reward)
                best_reward = episode_reward
                torch.save(
                    self.agent.q_network.state_dict(),
                    f"best_model_seed_{self.seed}.pth"
                )
                print(f"New best model saved (seed {self.seed}) with reward {best_reward}")
        return total_reward, loss_track

    def test(self, model_path, num_tests=5):

        # load trained weights
        self.agent.q_network.load_state_dict(torch.load(model_path))
        self.agent.q_network.eval()
        rewards = []
        for i in range(num_tests):
            seed = random.randint(0, 100000)
            env = MountainCarEnv(render_mode=None, seed=seed)
            state = env.reset()
            done = False
            total_reward = 0
            step_counter = 0

            while not done and step_counter < self.max_steps:
                # greedy action (no exploration)
                action = self.agent.select_action(state, epsilon=0)
                next_state, reward, done = env.step(action)

                state = next_state
                total_reward += reward
                step_counter += 1

            rewards.append(total_reward)

            print(f"Test {i + 1} | Seed {seed} | Reward {total_reward}")

        print("\nMean Test Reward:", np.mean(rewards))
        print("Std Reward:", np.std(rewards))

        return rewards

    # warm-up function
    def prefill_replay_memory(self, num_steps=20000):
        print(f"Prefilling replay buffer with {num_steps} random steps...")

        state = self.cart_pole_env.reset()

        for step in range(num_steps):
            action = self.cart_pole_env.action_space.sample()  # RANDOM action

            next_state, reward, done = self.cart_pole_env.step(action)

            self.agent.replay_memory.store(state, action, next_state, reward, done)

            if done:
                state = self.cart_pole_env.reset()
            else:
                state = next_state

        print("Replay memory prefill complete.")