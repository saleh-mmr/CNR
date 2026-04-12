import random
import numpy as np
import torch
from agents.agent import DQNAgent
from envs.cartpole import CartPoleEnv
from envs.mountaincar import MountainCarEnv
from envs.mycartpole import MyCartPoleEnv
import pandas as pd

class Trainer:
    def __init__(self, hyperparams, seed):
        # Load parameters
        self.discount_factor = hyperparams["discount_factor"]           # Bellman γ (future reward weight)
        self.batch_size = hyperparams["batch_size"]                     # Number of experiences per learning step
        self.max_episodes = hyperparams["max_episodes"]                 # number of episode for training or testing
        self.epsilon_max = hyperparams["epsilon_max"]                   # Initial exploration rate
        self.epsilon_min = hyperparams["epsilon_min"]                   # Minimum allowed epsilon
        self.epsilon_decay = hyperparams["epsilon_decay"]               # Exploration decay speed
        self.memory_capacity = hyperparams["memory_capacity"]           # Replay buffer size
        self.warmup_size = hyperparams["warmup_size"]                   # Number of random steps to fill replay memory before learning starts
        self.network_size = hyperparams["network_size"]                 # Number of neurons in hidden layers
        self.max_steps_per_episode = hyperparams["max_steps_per_episode"] # Max steps per episode to prevent infinite loops
        self.g_ap = hyperparams["G_ap"]                                  # Coefficient for Conductance ap
        self.g_p = hyperparams["G_p"]                                    # Coefficient for Conductance p
        self.g_bias = hyperparams["G_bias"]                              # Coefficient for Conductance bias
        self.seed = seed
        self.cartpole_env = CartPoleEnv(render_mode=None, seed=seed, max_steps=self.max_steps_per_episode)
        # self.mountaincar_env = MountainCarEnv(render_mode=None, seed=seed)
        self.mountaincar_env = MyCartPoleEnv(render_mode=None, seed=seed, max_steps=self.max_steps_per_episode)



        self.agent = DQNAgent(
            cartpole_env= self.cartpole_env,
            mountaincar_env= self.mountaincar_env,
            epsilon_max=self.epsilon_max,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            discount=self.discount_factor,
            memory_capacity=self.memory_capacity,
            network_size=self.network_size,
            g_ap=self.g_ap,
            g_p=self.g_p,
            g_bias=self.g_bias
        )

    def train(self):
        self.warmup_replay_memory(self.warmup_size)
        total_steps = 0
        total_rewards_in_episodes_cp = []
        total_rewards_in_episodes_mc = []
        window_size = 1
        training_logs = pd.DataFrame(columns=["Episode", "Reward_CP_0.5", "Reward_CP_0.7", "Epsilon"])
        details_logs = pd.DataFrame(columns=["batch_size", "epsilon_decay", "memory_size", "network_size", "warmup_size", "seed", "max_episodes", "max_steps_per_episode", "discount_factor", "G_ap_coefficient", "G_p_coefficient", "G_bias_coefficient"])
        details_logs.loc[len(details_logs)] = [self.batch_size, self.epsilon_decay, self.memory_capacity, self.network_size, self.warmup_size, self.seed, self.max_episodes, self.max_steps_per_episode, self.discount_factor, self.g_ap, self.g_p, self.g_bias]
        details_logs.to_csv(f"details_log.csv", index=False)



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
            while not done_cp and not done_mc:  # Limit steps to 100 per episode or until both environments are done
                step_counter += 1
                total_steps += 1

                # ---- CartPole ----
                if not done_cp:
                    action_cp = self.agent.select_action(state_cp, ap_index=0)
                    next_state_cp, reward_cp, done_cp = self.cartpole_env.step(action_cp)
                    self.agent.cartpole_memory.store(state_cp, action_cp, next_state_cp, reward_cp, done_cp)
                    state_cp = next_state_cp
                    episode_reward_cp += reward_cp
                    self.agent.learn(self.batch_size, ap_index=0)

                # ---- My Cart Pole ----
                if not done_mc:
                    action_mc = self.agent.select_action(state_mc, ap_index=1)
                    next_state_mc, reward_mc, done_mc = self.mountaincar_env.step(action_mc)
                    self.agent.mountaincar_memory.store(state_mc, action_mc, next_state_mc, reward_mc, done_mc)
                    state_mc = next_state_mc
                    episode_reward_mc += reward_mc
                    self.agent.learn(self.batch_size, ap_index=1)

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
            training_logs.loc[len(training_logs)] = [episode, episode_reward_cp, episode_reward_mc, self.agent.epsilon]

            # SAVE BEST MODEL For CP based on recent average reward
            if len(total_rewards_in_episodes_cp) >= window_size:
                recent_avg = np.mean(total_rewards_in_episodes_cp[-window_size:])
                if recent_avg >= self.max_steps_per_episode:
                    model_path = f"CP_best_model_seed_{self.seed}_{total_steps}.pth"
                    self.agent.weight_controller.load_weights(0)
                    torch.save(
                        self.agent.q_network.state_dict(),
                        model_path
                    )
                    print(f"1 Cartpole New best model saved (seed {self.seed}) with recent average reward {recent_avg:.2f} -> {model_path}")


            # SAVE BEST MODEL For MC based on recent average reward
            if len(total_rewards_in_episodes_mc) >= window_size:
               recent_avg = np.mean(total_rewards_in_episodes_mc[-window_size:])
               if recent_avg >= self.max_steps_per_episode:
                   model_path = f"MC_best_model_seed_{self.seed}_{total_steps}.pth"
                   self.agent.weight_controller.load_weights(1)
                   torch.save(
                       self.agent.q_network.state_dict(),
                       model_path
                   )
                   print(f"2 My Cart Pole New best model saved (seed {self.seed}) with recent average reward {recent_avg:.2f} -> {model_path}")


        training_logs.to_csv(f"training_log.csv", index=False)
        return total_rewards_in_episodes_cp, total_rewards_in_episodes_mc




    def test(self, model_path, num_tests=1000):

        # load trained weights
        self.agent.q_network.load_state_dict(torch.load(model_path))
        self.agent.q_network.eval()
        rewards = []
        tests = pd.DataFrame(columns=["test", "reward"])
        for test_num in range(num_tests):
            seed = random.randint(0, 4000)
            env = CartPoleEnv(render_mode=None, seed=seed)
            state = env.reset()
            done = False
            total_reward = 0
            step_counter = 0

            while not done:
                # greedy action (no exploration)
                action = self.agent.select_action_test(state)

                next_state, reward, done = env.step(action)
                # env.render()

                state = next_state
                total_reward += reward
                step_counter += 1

            rewards.append(total_reward)

            print(f"Test {test_num + 1} | Seed {seed} | Reward {total_reward}")
            tests.loc[len(tests)] = [test_num + 1 , total_reward]

        print("\nMean Test Reward:", np.mean(rewards))
        print("Std Reward:", np.std(rewards))
        tests.to_csv(f"tests_CP_0.5.csv", index=False)


        return rewards


    def warmup_replay_memory(self, num_steps):
        state_cp = self.cartpole_env.reset()
        state_mc = self.mountaincar_env.reset()
        for _ in range(num_steps):
            # random action for exploration
            action_cp = self.cartpole_env.action_space.sample()
            action_mc = self.mountaincar_env.action_space.sample()
            next_state_cp, reward_cp, done_cp = self.cartpole_env.step(action_cp)
            next_state_mc, reward_mc, done_mc = self.mountaincar_env.step(action_mc)
            self.agent.cartpole_memory.store(state_cp, action_cp, next_state_cp, reward_cp, done_cp)
            self.agent.mountaincar_memory.store(state_mc, action_mc, next_state_mc, reward_mc, done_mc)
            state_cp = self.cartpole_env.reset() if done_cp else next_state_cp
            state_mc = self.mountaincar_env.reset() if done_mc else next_state_mc