import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


class DeepMLP(nn.Module):
    def __init__(self, input_size=2, hidden_size=7, output_size=1, num_layers=10):
        super(DeepMLP, self).__init__()
        layers = []

        # First layer
        layers.append(nn.Linear(input_size, hidden_size, bias=False))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size, bias=False))

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size, bias=False))

        self.layers = nn.ModuleList(layers)
        # Leakey relu activation
        self.activation = lambda x: F.leaky_relu(x, negative_slope=0.01)

        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            nn.init.uniform_(layer.weight, a=-1.0, b=1.0)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x))
        return x



class SubMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SubMLP, self).__init__()
        layers = []

        # First layer
        layers.append(nn.Linear(input_size, hidden_size, bias=False))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size, bias=False))

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size, bias=False))

        self.layers = nn.ModuleList(layers)
        self.activation = lambda x: F.leaky_relu(x, negative_slope=0.01)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
            else:
                x = self.activation(x)  # output layer also has activation
        return x


# Load the original 9-layer large model
original_model = DeepMLP()
original_model.load_state_dict(torch.load("checkpoints/mlp_9layer_checkpoint.pth"))

# Instantiate two smaller MLPs
mlp1 = SubMLP(input_size=2, hidden_size=7, output_size=7, num_layers=5)
mlp2 = SubMLP(input_size=7, hidden_size=7, output_size=1, num_layers=5)

# Copy weights: MLP1 uses the first 4 layers of the original model
for i in range(5):
    mlp1.layers[i].weight.data = original_model.layers[i].weight.data.clone()
    print(f"layer {i} for MLP1: {mlp1.layers[i].weight.data}")

# MLP2 uses the remaining 4 layers
for i in range(5):
    mlp2.layers[i].weight.data = original_model.layers[i + 5].weight.data.clone()
    print(f"layer {i} for MLP2: {mlp2.layers[i].weight.data}")


# Create checkpoint directory
os.makedirs("checkpoints", exist_ok=True)

# Save weights
torch.save(mlp1.state_dict(), "checkpoints/mlp1_sub_checkpoint.pth")
torch.save(mlp2.state_dict(), "checkpoints/mlp2_sub_checkpoint.pth")
print("Saved mlp1 and mlp2 weights to checkpoints/")