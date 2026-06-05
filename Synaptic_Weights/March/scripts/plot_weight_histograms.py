import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
plt.rcParams.update({"font.size": 14})

base_dir = os.path.dirname(os.path.abspath(__file__))

# Give your weights folder here
folder = os.path.join(
    base_dir,
    "../weights/three_problem/run_2026-06-04_14-33-43"
)

step = 76329
layer = "FC.2"

weight_key = f"{layer}.weight"

# Number of histogram bins
bins = 50

# ---------------------------------------------------------
# Model paths
# ---------------------------------------------------------
path1 = os.path.join(folder, f"MC1_{step}.pth")
path2 = os.path.join(folder, f"MC2_{step}.pth")
path3 = os.path.join(folder, f"MC3_{step}.pth")

# ---------------------------------------------------------
# Read pole information from details_log.csv
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
# Load models
# ---------------------------------------------------------
state_dict1 = torch.load(path1, map_location="cpu")
state_dict2 = torch.load(path2, map_location="cpu")
state_dict3 = torch.load(path3, map_location="cpu")

# ---------------------------------------------------------
# Check selected layer exists
# ---------------------------------------------------------
for model_name, state_dict in [
    ("MC1", state_dict1),
    ("MC2", state_dict2),
    ("MC3", state_dict3),
]:
    if weight_key not in state_dict:
        raise KeyError(
            f"{weight_key} was not found in {model_name}.\n"
            f"Available keys are:\n{list(state_dict.keys())}"
        )

# ---------------------------------------------------------
# Extract and flatten weights
# ---------------------------------------------------------
w1 = state_dict1[weight_key].detach().cpu().numpy().flatten()
w2 = state_dict2[weight_key].detach().cpu().numpy().flatten()
w3 = state_dict3[weight_key].detach().cpu().numpy().flatten()

weights = [
    ("MC1", w1, pole_length_1, pole_mass_1),
    ("MC2", w2, pole_length_2, pole_mass_2),
    ("MC3", w3, pole_length_3, pole_mass_3),
]

# ---------------------------------------------------------
# Use the same x-axis range for fair comparison
# ---------------------------------------------------------
global_min = min(w1.min(), w2.min(), w3.min())
global_max = max(w1.max(), w2.max(), w3.max())

# ---------------------------------------------------------
# Plot 3 histograms as subplots
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharex=True, sharey=True)

for ax, (name, weight_values, pole_length, pole_mass) in zip(axes, weights):
    ax.hist(
        weight_values,
        bins=bins,
        range=(global_min, global_max),
        edgecolor="black",
        alpha=0.75
    )

    ax.set_title(
        f"{weight_key} Histogram - {name}\n"
        f"L={pole_length}, M={pole_mass}"
    )
    ax.set_xlabel("Weight value")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()