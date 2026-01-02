import numpy as np

from marl_meeting_task.src.env.meeting_gridworld import MeetingGridworldEnv


def main():
    # --------------------------------------------------
    # Create environment
    # --------------------------------------------------
    env = MeetingGridworldEnv(grid_size=5, n_agents=2, max_steps=50)

    print("Environment created")
    print("Initial agent positions (before reset):", env.agent_pos)
    print()

    # --------------------------------------------------
    # Reset with fixed seed
    # --------------------------------------------------
    obs, info = env.reset(seed=42)

    print("Environment reset with seed=42")
    print("Initial observations:")
    for agent_id, agent_obs in obs.items():
        print(f"  Agent {agent_id+1}: {agent_obs}")
    print("Goal position:", env.goal_pos)
    print()

    # --------------------------------------------------
    # RNG for action sampling (policy-side randomness)
    # --------------------------------------------------
    rng = np.random.default_rng(123)

    # --------------------------------------------------
    # Run a few random steps
    # --------------------------------------------------
    print("Starting random rollout")
    for step in range(5):
        # Sample random actions for each agent (dict keyed by agent id)
        actions = {i: rng.integers(0, 5) for i in range(env.n_agents)}

        obs, reward, terminated, truncated, info = env.step(actions)

        print(f"Step {step}")
        print("  Actions:", actions)
        print("  Agent positions:", env.agent_pos)
        print("  Reward:", reward)
        print("  Terminated:", terminated)
        print("  Truncated:", truncated)
        print()

        if terminated or truncated:
            print("Episode ended")
            break


if __name__ == "__main__":
    main()
