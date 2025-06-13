import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import random
from CreateDataset import CreateDataset
from autoencoder import AutoEncoder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import psutil
import logging
import os
import torch.nn.init as init
import argparse



parser = argparse.ArgumentParser(description="Train AutoEncoder for MLP Representation")
parser.add_argument('--cuda', type=int, default=0, help='CUDA device number to use (default: 0)')
args = parser.parse_args()

device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
print("Device: ", device)
log_file = open('training_log.txt', 'w')
logging.basicConfig(filename='memory_usage.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("training_plots", exist_ok=True)



def training_plot(all_loss, avg_loss, num_plot, epoch):
  avg_loss_np = [loss_item for loss_item in avg_loss]
  x_values = range(len(avg_loss_np))
  sampled_x_values = [i + 1 for i in range(0, len(x_values), num_plot)]
  sampled_avg_loss_np = avg_loss_np[::num_plot]

  plt.figure(figsize=(10, 5))
  plt.plot(sampled_x_values, sampled_avg_loss_np, marker='o', label='Average Loss')
  plt.title(f'Average Loss Record for Training Epoch {epoch}')
  plt.xlabel('Number of Batches')
  plt.ylabel('Loss')
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  filename = f'Training_average_loss_plot_{epoch}.png'
  save_path = os.path.join("training_plots", filename)
  plt.savefig(save_path)
  print(f"Training average loss plot saved as {filename}")



def plot_training_loss(training_avg_loss, save_path='all_training_loss.png'):
    epochs = range(1, 1 + len(training_avg_loss))  
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_avg_loss, marker='o', linestyle='-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training Loss vs. Epochs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training loss plot saved to {save_path}")




input_size = 2
output_size = 1
max_hidden_size = 7
output_layer_list = [1,2,3,4]

input_dim = 56
hidden_dim = 768
batch_size = 64

autoencoder = AutoEncoder(input_dim, hidden_dim, batch_size, max_hidden_size, device)
checkpoints = torch.load('../checkpoint/pretrain_ae.pth', map_location=device)
autoencoder.load_state_dict(checkpoints['model_state_dict'])

# Define optimizer
optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001)
optimizer.load_state_dict(checkpoints['optimizer_state_dict'])

autoencoder.train()


all_loss = []
avg_loss = []
all_valid = []
training_avg_loss = []


for epoch in range (51):

  # Retrain by controlling seed
  # torch.manual_seed(45) 
  # np.random.seed(45) 

  all_loss = []
  avg_loss = []

  for batch in range (50001):
    inputs, weights, outputs, masks, num_hidden_layers = CreateDataset(input_size, output_size, max_hidden_size, batch_size, device)

    # Add padding, prepare input MLPs for autoencoder
    num_pad = (4 - num_hidden_layers) * 8 + (4 - num_hidden_layers) + 3 * (4 - num_hidden_layers)
    padded_weights = F.pad(weights, (0, num_pad), mode="constant", value=0)
    padded_mask = F.pad(masks, (0, num_pad), mode="constant", value=0)
    stacked_input = torch.cat([padded_weights.unsqueeze(-1), padded_mask.unsqueeze(-1)], dim=-1)

    # Feed into autoencoder
    all_decoded, all_activation_funcs, all_activated_nodes, all_biases = autoencoder(stacked_input)

    # Calculate loss
    predictions_list = []
    for current_num_layer in output_layer_list:
      # Forward pass for decoded MLPs
      for i in range(current_num_layer):
        start_index = i * max_hidden_size
        end_index = min((i + 1) * max_hidden_size, all_decoded[current_num_layer-1].size(2))
                
        sliced_matrix = all_decoded[current_num_layer-1][:, :, start_index:end_index]
        num_padding = max_hidden_size - 2
        sliced_mask = all_activated_nodes[current_num_layer - 1][:, :, i]
        sliced_mask_expanded = sliced_mask.unsqueeze(1).expand(-1, inputs.size(1), -1)
        bias = all_biases[current_num_layer - 1][i]
        bias = bias.permute(0, 2, 1)

        if i == 0:
            sliced_matrix = sliced_matrix[:, :-num_padding, :] 
            z = torch.matmul(inputs, sliced_matrix) + bias
            act_weights = all_activation_funcs[current_num_layer - 1][i]
            act_weights = act_weights.unsqueeze(1)

            # Activation per neuron
            sigmoid_part = torch.sigmoid(z)
            tanh_part = torch.tanh(z)
            leaky_relu_part = F.leaky_relu(z, negative_slope=0.01)
            curr_result = (
                act_weights[..., 0] * sigmoid_part +
                act_weights[..., 1] * tanh_part +
                act_weights[..., 2] * leaky_relu_part
            )
            curr_result = curr_result * sliced_mask_expanded
            prev_result = curr_result

        else:
            z = torch.matmul(prev_result, sliced_matrix) + bias 
            act_weights = all_activation_funcs[current_num_layer - 1][i] 
            act_weights = act_weights.unsqueeze(1) 

            # Activation per neuron
            sigmoid_part = torch.sigmoid(z)
            tanh_part = torch.tanh(z)
            leaky_relu_part = F.leaky_relu(z, negative_slope=0.01)
            curr_result = (
                act_weights[..., 0] * sigmoid_part +
                act_weights[..., 1] * tanh_part +
                act_weights[..., 2] * leaky_relu_part
            )
            curr_result = curr_result * sliced_mask_expanded
            prev_result = curr_result

      sliced_matrix = all_decoded[current_num_layer-1][:, :, (current_num_layer) * max_hidden_size : (current_num_layer) * max_hidden_size + 1]
      bias = all_biases[current_num_layer - 1][-1].permute(0, 2, 1)
      bias = bias[:, :, 0].unsqueeze(1)
      prediction = torch.matmul(curr_result, sliced_matrix) + bias
      prediction = prediction.squeeze(-1)
      predictions_list.append(prediction)

    # Loss over 4 decoders
    all_prediction = torch.stack(predictions_list, dim=0)
    outputs = outputs.unsqueeze(0).squeeze(-1).repeat(4, 1, 1)
    square_loss = torch.abs(all_prediction - outputs) ** 2
    mse_vector = torch.mean(square_loss, dim=-1)
    min_loss, _ = torch.min(mse_vector, dim=0)
    total_loss = min_loss.mean()


    # zero the parameter gradients
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    all_loss.append(total_loss.item())
    curr_avg_loss = sum(all_loss) / len(all_loss)
    avg_loss.append(curr_avg_loss)


    if batch % 40 == 0:
      print(f"iteration loss, {epoch} epoch, {batch} batch: ", all_loss[batch])
      print(f"average loss, {epoch} epoch, {batch} batch: ", avg_loss[batch])
      print()


    if epoch % 5 == 0 and batch % 200 == 0 and batch > 0:
      training_plot(all_loss, avg_loss, (batch // 100), epoch)


    if batch % 5000 == 0 and batch > 0:
      # Check and print memory usage
      logging.info(f"Epoch {epoch}, Batch {batch}: GPU Memory - Allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB, Cached: {torch.cuda.memory_reserved(device)/1e9:.2f} GB")
      logging.info(f"Epoch {epoch}, Batch {batch}: CPU Memory - Used: {psutil.Process(os.getpid()).memory_info().rss / 1e9:.2f} GB")

  if epoch % 2 == 0 and epoch != 0:
    torch.save({
        'model_state_dict': autoencoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"checkpoints/{epoch}_more_further_{batch}.pth")

  training_avg_loss.append(avg_loss[-1])
  plot_training_loss(training_avg_loss)







