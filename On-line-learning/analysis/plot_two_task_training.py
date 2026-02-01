import pickle
import matplotlib.pyplot as plt


def main():
    with open("two_task_log.pkl", "rb") as f:
        logger = pickle.load(f)

    data = logger.to_numpy()

    steps = data["step"]

    # -------- Plot 1: weights vs training --------
    plt.figure()
    plt.plot(steps, data["w_fl"], label="W (FrozenLake)")
    plt.plot(steps, data["w_cl"], label="W (CliffWalking)")
    plt.xlabel("Global training step")
    plt.ylabel("Weight value")
    plt.title("Memristor weights vs training (FL vs CL)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # -------- Plot 2: episode progress --------
    plt.figure()
    plt.plot(steps, data["ep_fl"], label="Episodes completed (FrozenLake)")
    plt.plot(steps, data["ep_cl"], label="Episodes completed (CliffWalking)")
    plt.xlabel("Global training step")
    plt.ylabel("Episodes completed")
    plt.title("Episode progress vs training")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
