import matplotlib.pyplot as plt
import numpy as np

from logging_helper.pulse_logger import PulseLogger


def plot_w_vs_x(pulse_logger: PulseLogger):
    data = pulse_logger.to_numpy()

    x_plus = data["x_plus"]
    weight = data["weight"]

    # Remove duplicate x values (many steps may have same x)
    # Keep the last weight for each x
    unique_x = {}
    for x, w in zip(x_plus, weight):
        unique_x[x] = w

    xs = np.array(sorted(unique_x.keys()))
    ws = np.array([unique_x[x] for x in xs])

    plt.figure()
    plt.plot(xs, ws)
    plt.xlabel("Pulse index x_plus")
    plt.ylabel("Weight W")
    plt.title("Memristor synaptic weight vs pulse index (FrozenLake)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Load the logger object here however you saved it
    # Example: using pickle
    import pickle

    with open("pulse_log.pkl", "rb") as f:
        pulse_logger = pickle.load(f)

    plot_w_vs_x(pulse_logger)
