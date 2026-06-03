import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd  # Added to read the CSV log file

# Increase font size for all text elements
plt.rcParams.update({'font.size': 16})

# Paths to your models
base_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(base_dir, "three_problems/run_2026-06-03_22-18-06")
step = 109629
path1 = os.path.join(folder, f"MC1_{step}.pth")
path2 = os.path.join(folder, f"MC2_{step}.pth")

# ---------------------------------------------------------
# Read pole lengths dynamically from details_log.csv
# ---------------------------------------------------------
log_path = os.path.join(folder, "details_log.csv")
df_log = pd.read_csv(log_path)

# Extract the values from the first row of the log
pole_length_1 = df_log["CP_pole_length_1"].iloc[0]
pole_length_2 = df_log["CP_pole_length_2"].iloc[0]
pole_length_3 = df_log["CP_pole_length_3"].iloc[0]

# Load weights
state_dict1 = torch.load(path1, map_location="cpu")
state_dict2 = torch.load(path2, map_location="cpu")

# ---------------------------------------------------------
# 1. FC.2 Weight Heatmaps
# ---------------------------------------------------------
# Extract FC.2.weight
w1 = state_dict1["FC.4.weight"].detach().cpu().numpy()
w2 = state_dict2["FC.4.weight"].detach().cpu().numpy()

max_abs = max(
    abs(w1.min()), abs(w1.max()),
    abs(w2.min()), abs(w2.max())
)

plt.figure(figsize=(14, 6))

# Plot 1
plt.subplot(1, 2, 1)
sns.heatmap(w1, center=0, vmin=-max_abs, vmax=max_abs,
            xticklabels=5, yticklabels=5)
plt.title(f"FC.2.weight - MC1 (L={pole_length_1})")
plt.xticks(rotation=0)  # text stands upright
plt.yticks(rotation=0)  # text stands upright

# Plot 2
plt.subplot(1, 2, 2)
sns.heatmap(w2, center=0, vmin=-max_abs, vmax=max_abs,
            xticklabels=5, yticklabels=5)
plt.title(f"FC.2.weight - MC2 (L={pole_length_2})")
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 2. FC.2 Weight Difference
# ---------------------------------------------------------
diff = w1 - w2
max_abs_diff = np.max(np.abs(diff))

plt.figure(figsize=(8, 6))
sns.heatmap(diff, cmap="seismic", center=0,
            vmin=-max_abs_diff, vmax=max_abs_diff,
            xticklabels=5, yticklabels=5)
plt.title(f"Difference in FC.2.weight (L={pole_length_1} - L={pole_length_2})")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.show()

# ---------------------------------------------------------
# 3. FC.2 Bias Heatmaps
# ---------------------------------------------------------
# Extract bias
b1 = state_dict1["FC.2.bias"].detach().cpu().numpy().reshape(1, -1)
b2 = state_dict2["FC.2.bias"].detach().cpu().numpy().reshape(1, -1)

# Shared color scale
max_abs_b = max(
    abs(b1.min()), abs(b1.max()),
    abs(b2.min()), abs(b2.max())
)

plt.figure(figsize=(14, 4))

plt.subplot(1, 2, 1)
sns.heatmap(b1, cmap="seismic", center=0,
            vmin=-max_abs_b, vmax=max_abs_b,
            xticklabels=5, cbar=True)
plt.title(f"FC.2.bias - CartPole (L={pole_length_1})")
plt.xticks(rotation=0)
plt.yticks([]) # Biases only have 1 row, so we hide y-ticks

plt.subplot(1, 2, 2)
sns.heatmap(b2, cmap="seismic", center=0,
            vmin=-max_abs_b, vmax=max_abs_b,
            xticklabels=5, cbar=True)
plt.title(f"FC.2.bias - CartPole (L={pole_length_2})")
plt.xticks(rotation=0)
plt.yticks([])

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 4. FC.2 Bias Difference
# ---------------------------------------------------------
# Difference
diff_b = b1 - b2
max_abs_diff_b = np.max(np.abs(diff_b))

plt.figure(figsize=(10, 3))
sns.heatmap(diff_b, cmap="seismic", center=0,
            vmin=-max_abs_diff_b, vmax=max_abs_diff_b,
            xticklabels=5, cbar=True)
plt.title(f"Bias Difference (L={pole_length_1} - L={pole_length_2})")
plt.xticks(rotation=0)
plt.yticks([])
plt.show()