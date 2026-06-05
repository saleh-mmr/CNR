import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
plt.rcParams.update({"font.size": 14})
base_dir = os.path.dirname(os.path.abspath(__file__))

folder = os.path.join(base_dir, "../weights/three_problem/run_2026-06-04_14-33-43")
step = 76329
layer = "FC.2"

weight_key = f"{layer}.weight"
bias_key = f"{layer}.bias"

# ---------------------------------------------------------
# Paths to your 3 models
# ---------------------------------------------------------
path1 = os.path.join(folder, f"MC1_{step}.pth")
path2 = os.path.join(folder, f"MC2_{step}.pth")
path3 = os.path.join(folder, f"MC3_{step}.pth")

# ---------------------------------------------------------
# Read pole lengths and masses dynamically from details_log.csv
# ---------------------------------------------------------
log_path = os.path.join(folder, "details_log.csv")
df_log = pd.read_csv(log_path)

pole_length_1 = df_log["CP_pole_length_1"].iloc[0]
pole_length_2 = df_log["CP_pole_length_2"].iloc[0]
pole_length_3 = df_log["CP_pole_length_3"].iloc[0]

pole_mass_1 = df_log["CP_pole_mass_1"].iloc[0]
pole_mass_2 = df_log["CP_pole_mass_2"].iloc[0]
pole_mass_3 = df_log["CP_pole_mass_3"].iloc[0]

# ---------------------------------------------------------
# Load model weights
# ---------------------------------------------------------
state_dict1 = torch.load(path1, map_location="cpu")
state_dict2 = torch.load(path2, map_location="cpu")
state_dict3 = torch.load(path3, map_location="cpu")

# ---------------------------------------------------------
# Check that selected layer exists
# ---------------------------------------------------------
for key in [weight_key, bias_key]:
    if key not in state_dict1:
        raise KeyError(
            f"{key} was not found in the model.\n"
            f"Available keys are:\n{list(state_dict1.keys())}"
        )

# ---------------------------------------------------------
# Extract selected layer weights and biases
# ---------------------------------------------------------
w1 = state_dict1[weight_key].detach().cpu().numpy()
w2 = state_dict2[weight_key].detach().cpu().numpy()
w3 = state_dict3[weight_key].detach().cpu().numpy()

b1 = state_dict1[bias_key].detach().cpu().numpy().reshape(1, -1)
b2 = state_dict2[bias_key].detach().cpu().numpy().reshape(1, -1)
b3 = state_dict3[bias_key].detach().cpu().numpy().reshape(1, -1)

models = [
    ("MC1", w1, b1, pole_length_1, pole_mass_1),
    ("MC2", w2, b2, pole_length_2, pole_mass_2),
    ("MC3", w3, b3, pole_length_3, pole_mass_3),
]

# ---------------------------------------------------------
# 1. Weight heatmaps for MC1, MC2, MC3
# ---------------------------------------------------------
max_abs_w = max(
    np.max(np.abs(w1)),
    np.max(np.abs(w2)),
    np.max(np.abs(w3))
)

fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharey=True)

for ax, (name, weights, bias, pole_length, pole_mass) in zip(axes, models):
    sns.heatmap(
        weights,
        ax=ax,
        cmap="seismic",
        center=0,
        vmin=-max_abs_w,
        vmax=max_abs_w,
        xticklabels=5,
        yticklabels=5,
        cbar=True
    )

    ax.set_title(f"{weight_key} - {name}\nL={pole_length}, M={pole_mass}")
    ax.set_xlabel("Input neuron index")
    ax.set_ylabel("Output neuron index")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 2. Bias heatmaps for MC1, MC2, MC3
# ---------------------------------------------------------
max_abs_b = max(
    np.max(np.abs(b1)),
    np.max(np.abs(b2)),
    np.max(np.abs(b3))
)

fig, axes = plt.subplots(1, 3, figsize=(22, 4), sharey=True)

for ax, (name, weights, bias, pole_length, pole_mass) in zip(axes, models):
    sns.heatmap(
        bias,
        ax=ax,
        cmap="seismic",
        center=0,
        vmin=-max_abs_b,
        vmax=max_abs_b,
        xticklabels=5,
        yticklabels=False,
        cbar=True
    )

    ax.set_title(f"{bias_key} - {name}\nL={pole_length}, M={pole_mass}")
    ax.set_xlabel("Bias neuron index")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 3. Difference of weights
# ---------------------------------------------------------
weight_differences = [
    ("MC1 - MC2", w1 - w2, pole_length_1, pole_length_2),
    ("MC1 - MC3", w1 - w3, pole_length_1, pole_length_3),
    ("MC2 - MC3", w2 - w3, pole_length_2, pole_length_3),
]

max_abs_diff_w = max(np.max(np.abs(diff)) for _, diff, _, _ in weight_differences)

fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharey=True)

for ax, (title, diff, length_a, length_b) in zip(axes, weight_differences):
    sns.heatmap(
        diff,
        ax=ax,
        cmap="seismic",
        center=0,
        vmin=-max_abs_diff_w,
        vmax=max_abs_diff_w,
        xticklabels=5,
        yticklabels=5,
        cbar=True
    )

    ax.set_title(f"{weight_key} Difference\n{title}\nL={length_a} - L={length_b}")
    ax.set_xlabel("Input neuron index")
    ax.set_ylabel("Output neuron index")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 4. Difference of biases
# ---------------------------------------------------------
bias_differences = [
    ("MC1 - MC2", b1 - b2, pole_length_1, pole_length_2),
    ("MC1 - MC3", b1 - b3, pole_length_1, pole_length_3),
    ("MC2 - MC3", b2 - b3, pole_length_2, pole_length_3),
]

max_abs_diff_b = max(np.max(np.abs(diff)) for _, diff, _, _ in bias_differences)

fig, axes = plt.subplots(1, 3, figsize=(22, 4), sharey=True)

for ax, (title, diff, length_a, length_b) in zip(axes, bias_differences):
    sns.heatmap(
        diff,
        ax=ax,
        cmap="seismic",
        center=0,
        vmin=-max_abs_diff_b,
        vmax=max_abs_diff_b,
        xticklabels=5,
        yticklabels=False,
        cbar=True
    )

    ax.set_title(f"{bias_key} Difference\n{title}\nL={length_a} - L={length_b}")
    ax.set_xlabel("Bias neuron index")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.show()