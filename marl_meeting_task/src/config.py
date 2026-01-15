import os
import gc
import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gc.collect()
torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Seed everything for reproducible results
seed = 2025
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Default MAPPO hyperparameters (can be overridden in scripts)
mappo_defaults = {
    'lr_actor': 3e-4,
    'lr_critic': 3e-4,
    'clip_eps': 0.2,
    'value_coef': 0.5,
    'entropy_coef': 0.01,
    'ppo_epochs': 4,
    'mini_batch_size': 64,
    'gae_lambda': 0.95,
    'max_grad_norm': 0.5,
    'use_centralized_critic': True,
    'share_actor': True,
    'rollout_length': 128,
    'device': device,
}
