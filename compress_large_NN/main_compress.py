import torch
import torch.nn as nn
import torch.nn.functional as F
from autoencoder import AutoEncoder
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

class SubMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SubMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size, bias=False))
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
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



max_hidden_size = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load submlp1
mlp1 = SubMLP(input_size=2, hidden_size=7, output_size=7, num_layers=5).to(device)
mlp1.load_state_dict(torch.load("checkpoints/mlp1_sub_checkpoint.pth"))
mlp1.eval()

# Load submlp2
mlp2 = SubMLP(input_size=7, hidden_size=7, output_size=1, num_layers=5).to(device)
mlp2.load_state_dict(torch.load("checkpoints/mlp2_sub_checkpoint.pth"))
mlp2.eval()




data_path = "checkpoints/mlp_io_data.npz"
data = np.load(data_path)
X = data['inputs']
Y = data['outputs'] 
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)

# Feed into MLP1
with torch.no_grad():
    Y_mlp1 = mlp1(X_tensor)

z_values = Y_mlp1[:, 0].cpu().numpy()


with torch.no_grad():
    final_z = mlp2(Y_mlp1)
    final_z = final_z.cpu().numpy()









# Generate input for MLP1
processed_weights = []
processed_masks = []
for i, layer in enumerate(mlp1.layers):
    W = layer.weight.data.T 
    mask = torch.ones_like(W)

    if i == 0:
        pad_rows = 7 - W.shape[0]
        if pad_rows > 0:
            padding = torch.zeros((pad_rows, W.shape[1]), device=W.device)
            mask_pad = torch.zeros_like(padding)
            W = torch.cat([W, padding], dim=0)
            mask = torch.cat([mask, mask_pad], dim=0)

    processed_weights.append(W)
    processed_masks.append(mask)
final_matrix_mlp1 = torch.cat(processed_weights, dim=1)
final_mask_mlp1 = torch.cat(processed_masks, dim=1)
stacked_input_mlp1 = torch.stack([final_matrix_mlp1, final_mask_mlp1], dim=-1).unsqueeze(0)




# Generate input for MLP2
processed_weights_mlp2 = []
processed_masks_mlp2 = []
for i, layer in enumerate(mlp2.layers):
    W = layer.weight.data.T
    mask = torch.ones_like(W)
    if i == len(mlp2.layers) - 1:
        pad_cols = 7 - W.shape[1]
        if pad_cols > 0:
            padding = torch.zeros((W.shape[0], pad_cols), device=W.device)
            mask_pad = torch.zeros_like(padding)
            W = torch.cat([W, padding], dim=1)
            mask = torch.cat([mask, mask_pad], dim=1)
    processed_weights_mlp2.append(W)
    processed_masks_mlp2.append(mask)
final_matrix_mlp2 = torch.cat(processed_weights_mlp2, dim=1)
final_mask_mlp2 = torch.cat(processed_masks_mlp2, dim=1)
stacked_input_mlp2 = torch.stack([final_matrix_mlp2, final_mask_mlp2], dim=-1).unsqueeze(0)









# Train First subMLP
max_hidden_size = 7
current_num_layer = 2
input_dim = 35
hidden_dim = 768
batch_size = 1

# Load autoencoder
autoencoder = AutoEncoder(input_dim, hidden_dim, batch_size, max_hidden_size, device)
checkpoints = torch.load('checkpoints/100_batch_10000.pth', map_location=device)
autoencoder.load_state_dict(checkpoints['model_state_dict'])
autoencoder.eval()
for param in autoencoder.parameters():
    param.requires_grad = False

input_size = 2
output_size = 7
z_size = hidden_dim 
z = torch.randn(1, max_hidden_size, hidden_dim, device=device, requires_grad=True)
optimizer = optim.Adam([{'params': z, 'lr': 0.1}])

# Load first sub_MLP
data_path = "checkpoints/Y_mlp1_checkpoint.pth"
checkpoint = torch.load(data_path, map_location=device)
outputs = checkpoint['Y_mlp1'].to(device)

# Train first subMLP
best_loss = float("inf")
best_weights = None
for epoch in range (8001):
    decoder_out = autoencoder.decoder_7_23(inputs_embeds=z).last_hidden_state
    out_all = autoencoder.final_layer_7_23(decoder_out).view(batch_size, autoencoder.max_hidden_size, -1)
    reconstructed_weights = out_all[:, :, :21]

    for i in range(current_num_layer):

        start_index = i * max_hidden_size
        end_index = min((i + 1) * max_hidden_size, reconstructed_weights.size(2))
        
        sliced_matrix = reconstructed_weights[:, :, start_index:end_index]
        num_padding = max_hidden_size - input_size

        if i == 0:
            if num_padding > 0:
                sliced_matrix = sliced_matrix[:, :-num_padding, :] 
            # print(sliced_matrix.shape)
            prev_result = torch.nn.functional.leaky_relu(torch.matmul(X_tensor, sliced_matrix), negative_slope=0.01) 
            curr_result = prev_result
        else:
            curr_result = torch.nn.functional.leaky_relu(torch.matmul(prev_result, sliced_matrix), negative_slope=0.01) 
            prev_result = curr_result

    sliced_matrix = reconstructed_weights[:, :, (current_num_layer) * max_hidden_size : (current_num_layer) * max_hidden_size + output_size]
    prediction = torch.nn.functional.leaky_relu(torch.matmul(curr_result, sliced_matrix), negative_slope=0.01)  
    prediction = prediction.squeeze(0)
    loss = F.mse_loss(prediction, outputs)


    # Train the first subMLP
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if loss < best_loss:
        best_loss = loss
        best_weights = reconstructed_weights.detach().clone()

    if epoch % 200 == 0:
        print(f"[Epoch {epoch}] Training Loss: {loss.item():.6f}")


# After getting the best
with torch.no_grad():
    curr_result = None
    for i in range(current_num_layer):
        start_index = i * max_hidden_size
        end_index = min((i + 1) * max_hidden_size, best_weights.size(2))

        sliced_matrix = best_weights[:, :, start_index:end_index]  # [batch_size, num_samples, hidden_size]
        num_padding = max_hidden_size - input_size

        if i == 0:
            if num_padding > 0:
                sliced_matrix = sliced_matrix[:, :-num_padding, :]  
            prev_result = F.leaky_relu(torch.matmul(X_tensor, sliced_matrix), negative_slope=0.01)
            curr_result = prev_result
        else:
            curr_result = F.leaky_relu(torch.matmul(prev_result, sliced_matrix), negative_slope=0.01)
            prev_result = curr_result

    sliced_matrix = best_weights[:, :, (current_num_layer) * max_hidden_size : (current_num_layer) * max_hidden_size + output_size]
    final_prediction = F.leaky_relu(torch.matmul(curr_result, sliced_matrix), negative_slope=0.01)  # [N, output_size]
    final_prediction = final_prediction.squeeze(0)  # Shape: [N, output_size]

# Saved the compressed subMLP1
save_path = "checkpoints/ablation_MLP1_predictions.pth"
torch.save({
    'final_prediction': final_prediction.cpu(),
    'best_weights': best_weights.cpu()
}, save_path)

















# Train second subMLP
max_hidden_size = 7
current_num_layer = 2
input_dim = 35
hidden_dim = 768
batch_size = 1


input_size = 7
output_size = 1

autoencoder = AutoEncoder(input_dim, hidden_dim, batch_size, max_hidden_size)
checkpoints = torch.load('checkpoints/100_batch_10000.pth')
autoencoder.load_state_dict(checkpoints['model_state_dict'])
autoencoder.eval()
for param in autoencoder.parameters():
    param.requires_grad = False

z_size = hidden_dim 
z = torch.randn(1, max_hidden_size, hidden_dim, device=device, requires_grad=True)
optimizer = optim.Adam([{'params': z, 'lr': 0.1}])
# output of MLP1 as input of MLP2
data_path = "checkpoints/ablation_MLP1_predictions.pth"
checkpoint = torch.load(data_path, map_location=device)
inputs = checkpoint['final_prediction'].to(device)

# Train the second subMLP
best_loss = float("inf")
best_weights = None
for epoch in range (8001):
    decoder_out = autoencoder.decoder_7_23(inputs_embeds=z).last_hidden_state
    out_all = autoencoder.final_layer_7_23(decoder_out).view(batch_size, autoencoder.max_hidden_size, -1)
    reconstructed_weights = out_all[:, :, :21]

    for i in range(current_num_layer):

        start_index = i * max_hidden_size
        end_index = min((i + 1) * max_hidden_size, reconstructed_weights.size(2))
        
        sliced_matrix = reconstructed_weights[:, :, start_index:end_index]  # [batch_size, num_samples, hidden_size]
        num_padding = max_hidden_size - input_size

        if i == 0:
            if num_padding > 0:
                sliced_matrix = sliced_matrix[:, :-num_padding, :]  # [batch_size, max_hidden_size - 3, hidden_size]
            prev_result = torch.nn.functional.leaky_relu(torch.matmul(inputs, sliced_matrix), negative_slope=0.01)  # [batch_size, max_hidden_size - 3, hidden_size]
            curr_result = prev_result
        else:
            curr_result = torch.nn.functional.leaky_relu(torch.matmul(prev_result, sliced_matrix), negative_slope=0.01)  # [batch_size, num_samples, hidden_size]
            prev_result = curr_result

    sliced_matrix = reconstructed_weights[:, :, (current_num_layer) * max_hidden_size : (current_num_layer) * max_hidden_size + output_size]
    prediction = torch.nn.functional.leaky_relu(torch.matmul(curr_result, sliced_matrix), negative_slope=0.01)   # [batch_size, num_samples]
    prediction = prediction.squeeze(0)

    loss = F.mse_loss(prediction, Y_tensor)

    # Train the second subMLP
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if loss < best_loss:
        best_loss = loss
        best_weights = reconstructed_weights.detach().clone()

    if epoch % 200 == 0:
        print(f"[Epoch {epoch}] Training Loss: {loss.item():.6f}")


# After getting the best
with torch.no_grad():
    curr_result = None
    for i in range(current_num_layer):
        start_index = i * max_hidden_size
        end_index = min((i + 1) * max_hidden_size, best_weights.size(2))

        sliced_matrix = best_weights[:, :, start_index:end_index]  # [batch_size, num_samples, hidden_size]
        num_padding = max_hidden_size - input_size

        if i == 0:
            if num_padding > 0:
                sliced_matrix = sliced_matrix[:, :-num_padding, :]  
            prev_result = F.leaky_relu(torch.matmul(inputs, sliced_matrix), negative_slope=0.01)
            curr_result = prev_result
        else:
            curr_result = F.leaky_relu(torch.matmul(prev_result, sliced_matrix), negative_slope=0.01)
            prev_result = curr_result

    sliced_matrix = best_weights[:, :, (current_num_layer) * max_hidden_size : (current_num_layer) * max_hidden_size + output_size]
    final_prediction = F.leaky_relu(torch.matmul(curr_result, sliced_matrix), negative_slope=0.01)  # [N, output_size]
    final_prediction = final_prediction.squeeze(0)  # Shape: [N, output_size]

save_path = "checkpoints/ablation_MLP2_predictions.pth"
torch.save({
    'final_prediction': final_prediction.cpu(),
    'best_weights': best_weights.cpu()
}, save_path)
print("Saved!")
