import random
import numpy as np
import torch
from agents.agent import DQNAgent
from envs.cartpole import MyCartPoleEnv
from envs.mountaincar import MountainCarEnv


def warmup_env(env, memory, num_steps):
    state = env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()
        next_state, reward, done = env.step(action)
        memory.store(state, action, next_state, reward, done)
        state = env.reset() if done else next_state


class Trainer:
    def __init__(self, hyperparams, seed):
        # Load parameters
        self.discount_factor = hyperparams["discount_factor"]           # Bellman γ (future reward weight)
        self.batch_size = hyperparams["batch_size"]                     # Number of experiences per learning step
        self.max_episodes = hyperparams["max_episodes"]                 # number of episode for training or testing
        self.max_steps = hyperparams["max_steps"]                       # Episode timeout
        self.epsilon_max = hyperparams["epsilon_max"]                   # Initial exploration rate
        self.epsilon_min = hyperparams["epsilon_min"]                   # Minimum allowed epsilon
        self.epsilon_decay = hyperparams["epsilon_decay"]               # Exploration decay speed
        self.memory_capacity = hyperparams["memory_capacity"]           # Replay buffer size
        self.seed = seed
        self.cartpole_env = MyCartPoleEnv(render_mode=None, seed=seed, length=0.5)
        self.mountaincar_env = MountainCarEnv(render_mode=None, seed=seed)


        self.agent = DQNAgent(
            cartpole_env= self.cartpole_env,
            mountaincar_env= self.mountaincar_env,
            epsilon_max=self.epsilon_max,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            discount=self.discount_factor,
            memory_capacity=self.memory_capacity,
        )

    def train(self):
        warmup_env(self.cartpole_env, self.agent.cartpole_memory, 3000)
        warmup_env(self.mountaincar_env, self.agent.mountaincar_memory, 3000)

        total_steps = 0
        window_size = 5
        best_so_far = -float("inf")
        total_rewards_in_episodes_cp = []
        total_rewards_in_episodes_mc = []

        for episode in range(1, self.max_episodes + 1):
            # Initial observation from environment
            state_cp = self.cartpole_env.reset()
            state_mc = self.mountaincar_env.reset()
            # Flags to track episode completion for each environment
            done_cp = False
            done_mc = False
            # Total reward accumulated in this episode each environment (for logging)
            episode_reward_cp = 0
            episode_reward_mc = 0
            step_counter = 0                # Step counter inside episode
            while not done_cp and not done_mc and step_counter < self.max_steps:
                step_counter += 1
                total_steps += 1
                # For each environment, if it's not done, select action, step, store experience, and accumulate reward
                if not done_cp:
                    action_cp = self.agent.select_action(state_cp)
                    # Step in the environment and get next state, reward, and done flag
                    next_state_cp, reward_cp, done_cp = self.cartpole_env.step(action_cp)
                    # Store experience in the corresponding replay memory
                    self.agent.cartpole_memory.store(state_cp, action_cp, next_state_cp, reward_cp, done_cp)
                    state_cp = next_state_cp
                    episode_reward_cp += reward_cp
                    self.agent.learn(self.batch_size, 0)

                if not done_mc:
                    action_mc = self.agent.select_action(state_mc)
                    # Step in the environment and get next state, reward, and done flag
                    next_state_mc, reward_mc, done_mc = self.mountaincar_env.step(action_mc)
                    # Store experience in the corresponding replay memory
                    self.agent.mountaincar_memory.store(state_mc, action_mc, next_state_mc, reward_mc, done_mc)
                    state_mc = next_state_mc
                    episode_reward_mc += reward_mc
                    self.agent.learn(self.batch_size, 1)

            total_rewards_in_episodes_cp.append(episode_reward_cp)
            total_rewards_in_episodes_mc.append(episode_reward_mc)
            # Update epsilon (step-based)
            self.agent.update_epsilon(total_steps)

            # Shows training progress in readable way
            print(
                f"Episode: {episode}, "
                f"Steps: {step_counter}, "
                f"CP_reward: {episode_reward_cp:.2f}, "
                f"MC_reward: {episode_reward_mc:.2f}, "
                f"Epsilon: {self.agent.epsilon:.2f}"
            )

            # SAVE BEST MODEL
            if len(total_rewards_in_episodes_cp) >= window_size:
                recent_avg = np.mean(total_rewards_in_episodes_cp[-window_size:])
                if recent_avg >= best_so_far:
                    best_so_far = recent_avg
                    model_path = f"best_model_seed_{self.seed}.pth"
                    torch.save(
                        self.agent.q_network.state_dict(),
                        model_path
                    )
                    print(
                        f"New best model saved (seed {self.seed}) with recent average reward {recent_avg:.2f} -> {model_path}")

        return total_rewards_in_episodes_cp, total_rewards_in_episodes_mc




    def test(self, model_path, num_tests=100):

        # load trained weights
        self.agent.q_network.load_state_dict(torch.load(model_path))
        self.agent.q_network.eval()
        rewards = []
        for test_num in range(num_tests):
            seed = random.randint(0, 3000)
            env = CartPoleEnv(render_mode=None, seed=seed)
            state = env.reset()
            done = False
            total_reward = 0
            step_counter = 0

            while not done and step_counter < self.max_steps:
                # greedy action (no exploration)
                action = self.agent.select_action(state, epsilon=0)

                next_state, reward, done = env.step(action)
                # env.render()

                state = next_state
                total_reward += reward
                step_counter += 1

            rewards.append(total_reward)

            print(f"Test {test_num + 1} | Seed {seed} | Reward {total_reward}")

        print("\nMean Test Reward:", np.mean(rewards))
        print("Std Reward:", np.std(rewards))

        return rewards

