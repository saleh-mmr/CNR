import torch

# Load the checkpoint
state_dict = torch.load("CP_best_model_seed_49_494.pth", map_location="cpu")

# Check available keys (optional but useful)
print(state_dict.keys())

# Access the value
value = state_dict['FC.2.weight'][0, 0].item()

print(value)