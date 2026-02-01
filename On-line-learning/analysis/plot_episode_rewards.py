import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_rewards(filename, title, ylabel):
    with open(filename, "rb") as f:
        logger = pickle.load(f)

    data = logger.to_numpy()
    ep = data["episode"]
    r = data["reward"]

    plt.figure()
    plt.plot(ep, r)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_running_average(filename, title, ylabel, window=100):
    with open(filename, "rb") as f:
        logger = pickle.load(f)

    data = logger.to_numpy()
    r = data["reward"]

    if len(r) >= window:
        r_avg = np.convolve(r, np.ones(window)/window, mode="valid")
        ep = np.arange(len(r_avg))
    else:
        r_avg = r
        ep = np.arange(len(r))

    plt.figure()
    plt.plot(ep, r_avg)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title + f" (running avg, window={window})")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_rewards(
        "episode_rewards_fl.pkl",
        title="FrozenLake: reward per episode",
        ylabel="Episode return (0 or 1)",
    )

    plot_running_average(
        "episode_rewards_fl.pkl",
        title="FrozenLake: success rate",
        ylabel="Average success",
        window=200,
    )

    plot_rewards(
        "episode_rewards_cl.pkl",
        title="CliffWalking: reward per episode",
        ylabel="Episode return (negative)",
    )

    plot_running_average(
        "episode_rewards_cl.pkl",
        title="CliffWalking: learning progress",
        ylabel="Average episode return",
        window=50,
    )
