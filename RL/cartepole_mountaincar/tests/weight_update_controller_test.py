import os
import sys
import numpy as np
import torch.nn
from torch import nn
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from controller.weight_controller import ManhattanWeightController



network = torch.nn.Sequential(
    torch.nn.Linear(1, 4),
    nn.LeakyReLU(negative_slope=0.01),
    torch.nn.Linear(4, 1)
)


criterion = nn.MSELoss()
controller = ManhattanWeightController(network)

# Data
x = np.arange(30).reshape(-1, 1)
a = 2
b = 1
y_true = a * x + b

x_tensor = torch.as_tensor(x, dtype=torch.float32)
y_tensor = torch.as_tensor(y_true, dtype=torch.float32)

# Training loop
max_epochs = 4000

def track_parameter(network, criterion, controller, x_tensor, y_tensor, max_epochs=500, tracked_param_index=(0, 0)):
    x_index = []
    bias_index = []
    g_ap = []
    g_bias = []
    weights = []
    losses = []

    for epoch in range(max_epochs):
        # Forward
        y_prediction = network(x_tensor)
        loss = criterion(y_prediction, y_tensor)
        # Backward
        network.zero_grad()
        loss.backward()
        # Manhattan update
        controller.step(ap_index=0)

        # track the states of the synapses corresponding to the tracked parameter index
        '''
        self.synapses[0]  → first layer weights
        self.synapses[1]  → first layer bias
        self.synapses[2]  → second layer weights
        self.synapses[3]  → second layer bias
        '''
        param, syn_array = controller.synapses[tracked_param_index[0]]
        traced_synapse = syn_array[tracked_param_index[1]].item()
        idx = traced_synapse.get_positive_crosspoint_state(0)[0]
        idb = traced_synapse.get_bias_crosspoint_state()[0]
        ap = traced_synapse.get_positive_crosspoint_conductance_ap(0)
        bias = traced_synapse.get_bias_crosspoint_conductance()
        weight = traced_synapse.weight(0)
        x_index.append(idx)
        bias_index.append(idb)
        g_ap.append(ap)
        g_bias.append(bias)
        weights.append(weight)
        losses.append(loss.item())

        current_loss = loss.item()
        print(f"Epoch {epoch:3d} | Loss: {current_loss:.6f}")

    return x_index, bias_index, g_ap, g_bias, weights, losses



x_index, bias_index, g_ap, g_bias, weight, loss = track_parameter(network, criterion, controller, x_tensor, y_tensor, max_epochs=max_epochs, tracked_param_index=(0, 0))


epochs = range(len(x_index))

# First Plot: x_index and bias_index
plt.figure()
plt.plot(x_index, label='x_index')
plt.plot(bias_index, label='bias_index')
plt.xlabel('Epoch')
plt.ylabel('Index Value')
plt.title('Tracked Index Parameters')
plt.legend()
plt.show()

# Second Plot: g_ap, g_bias, and weight
plt.figure()
plt.plot(epochs, g_ap, label='g_ap')
plt.plot(epochs, g_bias, label='g_bias')
plt.plot(epochs, weight, label='weight')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Tracked Gradients and Weight')
plt.legend()
plt.show()

# Second Plot: g_ap, g_bias, and weight
plt.figure()
plt.plot(epochs, loss, label='loss')  # fixed label
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.title('Loss over Epochs')
plt.legend()
plt.show()