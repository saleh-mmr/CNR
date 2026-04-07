import random
import numpy as np
import torch
from agents.agent import DQNAgent
from envs.cartpole import CartPoleEnv
from envs.mountaincar import MountainCarEnv
from envs.mycartpole import MyCartPoleEnv


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
        self.seed = seed
        self.cartpole_env = CartPoleEnv(render_mode=None, seed=seed)
        # self.mountaincar_env = MountainCarEnv(render_mode=None, seed=seed)
        self.mountaincar_env = MyCartPoleEnv(render_mode=None, seed=seed)



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
        self.warmup_replay_memory(100)
        total_steps = 0
        total_rewards_in_episodes_cp = []
        total_rewards_in_episodes_mc = []
        window_size = 20
        best_so_far_cp = -float("inf")
        best_so_far_mc = -float("inf")


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
            while not done_cp:  # Limit steps to 100 per episode or until both environments are done
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
                # if not done_mc:
                #     action_mc = self.agent.select_action(state_mc, ap_index=1)
                #     next_state_mc, reward_mc, done_mc = self.mountaincar_env.step(action_mc)
                #     self.agent.mountaincar_memory.store(state_mc, action_mc, next_state_mc, reward_mc, done_mc)
                #     state_mc = next_state_mc
                #     episode_reward_mc += reward_mc
                #     self.agent.learn(self.batch_size, ap_index=1)

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

            # SAVE BEST MODEL For CP based on recent average reward
            if len(total_rewards_in_episodes_cp) >= window_size:
                recent_avg = np.mean(total_rewards_in_episodes_cp[-window_size:])
                if recent_avg >= best_so_far_cp:
                    best_so_far_cp = recent_avg
                    model_path = f"CP_best_model_seed_{self.seed}_{total_steps}.pth"
                    self.agent.weight_controller.load_weights(0)
                    torch.save(
                        self.agent.q_network.state_dict(),
                        model_path
                    )
                    print(f"Cartpole New best model saved (seed {self.seed}) with recent average reward {recent_avg:.2f} -> {model_path}")


            # # SAVE BEST MODEL For MC based on recent average reward
            #if len(total_rewards_in_episodes_mc) >= window_size:
            #    recent_avg = np.mean(total_rewards_in_episodes_mc[-window_size:])
            #    if recent_avg >= best_so_far_mc:
            #        best_so_far_mc = recent_avg
            #        model_path = f"MC_best_model_seed_{self.seed}.pth"
            #        self.agent.weight_controller.load_weights(1)
            #        torch.save(
            #            self.agent.q_network.state_dict(),
            #            model_path
            #        )
            #        print(f"My Cart Pole New best model saved (seed {self.seed}) with recent average reward {recent_avg:.2f} -> {model_path}")


        return total_rewards_in_episodes_cp, total_rewards_in_episodes_mc




    def test(self, model_path, num_tests=500):

        # load trained weights
        self.agent.q_network.load_state_dict(torch.load(model_path))
        self.agent.q_network.eval()
        rewards = []
        for test_num in range(num_tests):
            seed = random.randint(0, 4000)
            env = MyCartPoleEnv(render_mode=None, seed=seed)
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

        print("\nMean Test Reward:", np.mean(rewards))
        print("Std Reward:", np.std(rewards))

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