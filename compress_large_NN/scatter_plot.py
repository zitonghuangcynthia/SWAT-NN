import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser(description="Compress large network")
parser.add_argument('--cuda', type=int, default=0, help='CUDA device number to use (default: 0)')
args = parser.parse_args()
device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

class MLP9HiddenLayers(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=7, output_dim=1):
        super(MLP9HiddenLayers, self).__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim, bias=False))

        # 9 Hidden layers
        for _ in range(8):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim, bias=False))

        self.layers = nn.ModuleList(layers)
        self.activation = lambda x: F.leaky_relu(x, negative_slope=0.01)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)  # 每一层后都有activation，包括output层
        return x

model = MLP9HiddenLayers().to(device)
checkpoint_path = "/home/lies_mlp/single_encoder_arch/transformer/3_3_ViT/3_3_simpler/2_I/large_model/checkpoints/mlp_9layer_checkpoint.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of weights (parameters) in MLP9HiddenLayers: {total_params}")



num_samples = 1000 
input_dim = 2 

X_random = np.random.uniform(low=-1.0, high=1.0, size=(num_samples, input_dim)).astype(np.float32)
X_tensor = torch.tensor(X_random, device=device)


# For original MLP
with torch.no_grad():
    y_original = model(X_tensor) 

y_original = y_original.squeeze(-1).cpu().numpy()










# For decoded
# Plot Overall Graph
checkpoint1 = torch.load("checkpoints/ablation_MLP1_predictions.pth", map_location=device)
best_weights1 = checkpoint1['best_weights']
print(best_weights1)
checkpoint2 = torch.load("checkpoints/ablation_MLP2_predictions.pth", map_location=device)
best_weights2 = checkpoint2['best_weights'] 
print(best_weights2)


non_zero_counts = []
current_num_layer = 2
max_hidden_size = 7
with torch.no_grad():
    input_size = 2
    output_size = 7
    curr_result = None
    for i in range(current_num_layer):
        start_index = i * max_hidden_size
        end_index = min((i + 1) * max_hidden_size, best_weights1.size(2))

        sliced_matrix = best_weights1[:, :, start_index:end_index]  # [batch_size, num_samples, hidden_size]
        num_padding = max_hidden_size - input_size

        if i == 0:
            if num_padding > 0:
                sliced_matrix = sliced_matrix[:, :-num_padding, :]  
            nonzero = (sliced_matrix.abs() > 1e-6).sum().item()
            non_zero_counts.append(nonzero)
            prev_result = F.leaky_relu(torch.matmul(X_tensor, sliced_matrix), negative_slope=0.01)
            curr_result = prev_result
        else:
            nonzero = (sliced_matrix.abs() > 1e-6).sum().item()
            non_zero_counts.append(nonzero)
            curr_result = F.leaky_relu(torch.matmul(prev_result, sliced_matrix), negative_slope=0.01)
            prev_result = curr_result

    # Final output layer
    nonzero = (sliced_matrix.abs() > 1e-6).sum().item()
    non_zero_counts.append(nonzero)
    sliced_matrix = best_weights1[:, :, (current_num_layer) * max_hidden_size : (current_num_layer) * max_hidden_size + output_size]
    inter_prediction = F.leaky_relu(torch.matmul(curr_result, sliced_matrix), negative_slope=0.01)  # [N, output_size]
    inter_prediction = inter_prediction.squeeze(0)  # Shape: [N, output_size]
    print("inter shape: ", inter_prediction.shape)
print("SubMLP1 sliced matrix non-zero counts per layer:", non_zero_counts)
    
non_zero_counts2 = []
with torch.no_grad():
    input_size = 7
    output_size = 1
    curr_result = None
    for i in range(current_num_layer):
        start_index = i * max_hidden_size
        end_index = min((i + 1) * max_hidden_size, best_weights2.size(2))

        sliced_matrix = best_weights2[:, :, start_index:end_index]  # [batch_size, num_samples, hidden_size]
        num_padding = max_hidden_size - input_size

        if i == 0:
            if num_padding > 0:
                sliced_matrix = sliced_matrix[:, :-num_padding, :]  
            nonzero = (sliced_matrix.abs() > 1e-6).sum().item()
            non_zero_counts2.append(nonzero)
            prev_result = F.leaky_relu(torch.matmul(inter_prediction, sliced_matrix), negative_slope=0.01)
            curr_result = prev_result
        else:
            nonzero = (sliced_matrix.abs() > 1e-6).sum().item()
            non_zero_counts2.append(nonzero)
            curr_result = F.leaky_relu(torch.matmul(prev_result, sliced_matrix), negative_slope=0.01)
            prev_result = curr_result

    # Final output layer
    sliced_matrix = best_weights2[:, :, (current_num_layer) * max_hidden_size : (current_num_layer) * max_hidden_size + output_size]
    final_prediction = F.leaky_relu(torch.matmul(curr_result, sliced_matrix), negative_slope=0.01)  # [N, output_size]
    final_prediction = final_prediction.squeeze(0)  # Shape: [N, output_size]
    nonzero = (sliced_matrix.abs() > 1e-6).sum().item()
    non_zero_counts2.append(nonzero)
    print("final prediction shape: ", final_prediction.shape)
print("SubMLP2 sliced matrix non-zero counts per layer:", non_zero_counts2)







# Plot
final_prediction = final_prediction.detach().cpu().numpy()

plt.figure(figsize=(6, 6))
plt.scatter(y_original, final_prediction, c='blue', alpha=0.5, edgecolors='k', s=20)

min_val = min(y_original.min(), final_prediction.min())
max_val = max(y_original.max(), final_prediction.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

plt.xlabel('Original MLP Output', fontsize=24)
plt.ylabel('Compressed MLP Output', fontsize=24)
# plt.title('Original vs Compressed Output', fontsize=24)
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig("ablation_original_vs_compressed_scatter.png")
plt.show()