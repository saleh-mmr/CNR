import torch.nn

network = torch.nn.Sequential(
    torch.nn.Linear(2, 4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 2)
)

# for name, param in network.named_parameters():
#     print(f"Parameter name: {name}: params:\n{param}")
#     print("\n\n")

for name in network.named_parameters():
    for i in range(name[1].numel()):
        print(f"Parameter name: {name[0]}[{i}]: value:\n{name[1].flatten()[i]}")
        print("\n\n")