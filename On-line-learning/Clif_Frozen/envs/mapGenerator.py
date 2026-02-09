import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
import numpy as np
from collections import deque
import utils.config as config

def generate_random_map_rect(height, width, p=0.8):
    rng = np.random.default_rng(config.seed)

    while True:
        grid = rng.choice(
            ["F", "H"],
            size=(height, width),
            p=[p, 1 - p]
        )

        grid[0, 0] = "S"
        grid[-1, -1] = "G"

        if is_solvable(grid):
            return ["".join(row) for row in grid]


def is_solvable(grid):
    h, w = grid.shape
    start = (0, 0)
    goal = (h - 1, w - 1)

    queue = deque([start])
    visited = set([start])

    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True

        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < h and
                0 <= nc < w and
                (nr, nc) not in visited and
                grid[nr, nc] != "H"
            ):
                visited.add((nr, nc))
                queue.append((nr, nc))

    return False

#
# # ======================
# # Usage
# # ======================
#
#
# custom_map = generate_random_map_rect(
#     height=4,
#     width=12,
#     p=0.8,
# )
#
# env = gym.make(
#     "FrozenLake-v1",
#     desc=custom_map,
#     is_slippery=False
# )
#
# env.reset(seed=config.seed)
#
# print("Generated 4x12 map:")
# for row in custom_map:
#     print(row)
