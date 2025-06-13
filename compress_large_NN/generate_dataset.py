import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="Compress large network")
parser.add_argument('--cuda', type=int, default=0, help='CUDA device number to use (default: 0)')
args = parser.parse_args()
device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# Define MLP
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

# Instantiate and Move to Device
model = DeepMLP().to(device)
model.eval()

# Generate Random Input
np.random.seed(42)
X = np.random.uniform(-1, 1, size=(1000, 2)).astype(np.float32)
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

# Get Predictions
with torch.no_grad():
    Y_tensor = model(X_tensor)
Y = Y_tensor.cpu().numpy()

# Save Input-Output Pair
os.makedirs("checkpoints", exist_ok=True)
np.savez("checkpoints/mlp_io_data.npz", inputs=X, outputs=Y)
print("Saved input-output pairs to mlp_io_data.npz")

# Save large network's checkpoint
torch.save(model.state_dict(), "checkpoints/mlp_9layer_checkpoint.pth")
print("Saved MLP weights to checkpoints/mlp_9layer_checkpoint.pth")