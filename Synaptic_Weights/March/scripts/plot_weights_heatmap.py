import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to your models
path1 = "CP_best_model_seed_49_5083.pth"
path2 = "MC_best_model_seed_49_5083.pth"

# Load weights
state_dict1 = torch.load(path1, map_location="cpu")
state_dict2 = torch.load(path2, map_location="cpu")

# Extract FC.0.weight
w1 = state_dict1["FC.4.weight"].detach().cpu().numpy()
w2 = state_dict2["FC.4.weight"].detach().cpu().numpy()


max_abs = max(
    abs(w1.min()), abs(w1.max()),
    abs(w2.min()), abs(w2.max())
)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(w1, center=0,
            vmin=-max_abs, vmax=max_abs)
plt.title("FC.4.weight - CartPole with pole length 0.5")

plt.subplot(1, 2, 2)
sns.heatmap(w2, center=0,
            vmin=-max_abs, vmax=max_abs)
plt.title("FC.4.weight - CartPole with pole length 0.7")

plt.tight_layout()
plt.show()


diff = w1 - w2
max_abs_diff = np.max(np.abs(diff))

plt.figure(figsize=(6,5))
sns.heatmap(diff, cmap="seismic", center=0,
            vmin=-max_abs_diff, vmax=max_abs_diff)
plt.title("Difference in FC.4.weight (Default CP - Custom CP)")
plt.show()