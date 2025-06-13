import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import random
from autoencoder import AutoEncoder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import psutil
import logging
import os
import pickle
import torch.optim as optim
import lib.CORNN as CORNN
import argparse



parser = argparse.ArgumentParser(description="Train AutoEncoder for MLP Representation")
parser.add_argument('--cuda', type=int, default=0, help='CUDA device number to use (default: 0)')
args = parser.parse_args()
device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
print("Device: ", device)


def parse_decoder_output(output, config, max_hidden_size, device=None):

    weights_blocks = []
    activations_list = []
    bias_list = []

    for layer_idx in range(len(config)):
        if layer_idx not in config:
            continue

        w_start, w_end = config[layer_idx]["weight_cols"]
        weights = output[:, :, w_start:w_end]
        weights_blocks.append(weights)

        bias = output[:, :, w_end:w_end + 1]
        bias_list.append(bias)

        if "act_cols" in config[layer_idx]:
            a_start, a_end = config[layer_idx]["act_cols"]
            act_logits = output[:, :, a_start:a_end]
            activations_list.append(act_logits)

    stacked_weights = torch.cat(weights_blocks, dim=2) 

    if "mask_cols" in config:
        m_start, m_end = config["mask_cols"]
        structure_mask = torch.sigmoid(output[:, :, m_start:m_end])
    else:
        structure_mask = None
    return stacked_weights, activations_list, structure_mask, bias_list

# Layer config mappings
layer_config_1HL = {
    0: {"weight_cols": (0, 7),  "act_cols": (8, 11)},
    1: {"weight_cols": (11, 18)},
    "mask_cols": (19, 20),
}

layer_config_2HL = {
    0: {"weight_cols": (0, 7),   "act_cols": (8, 11)},
    1: {"weight_cols": (11, 18), "act_cols": (19, 22)},
    2: {"weight_cols": (22, 29)},
    "mask_cols": (30, 32),
}

layer_config_3HL = {
    0: {"weight_cols": (0, 7),   "act_cols": (8, 11)},
    1: {"weight_cols": (11, 18), "act_cols": (19, 22)},
    2: {"weight_cols": (22, 29), "act_cols": (30, 33)},
    3: {"weight_cols": (33, 40)},
    "mask_cols": (41, 44),
}

layer_config_4HL = {
     0: {"weight_cols": (0, 7),   "act_cols": (8, 11)},
    1: {"weight_cols": (11, 18), "act_cols": (19, 22)},
    2: {"weight_cols": (22, 29), "act_cols": (30, 33)},
    3: {"weight_cols": (33, 40), "act_cols": (41, 44)},
    4: {"weight_cols": (44, 51)},
    "mask_cols": (52, 56),
}



def loss_fn_2(outputs, predictions, decoded_MLP, all_biases, t, num_layers):
    pred_loss = ((outputs - predictions) ** 2).mean()  # MSE Loss

    fixed_coeff = 0.0001
    smooth_threshold = 0.01 * torch.sigmoid(20 * (torch.abs(decoded_MLP) - t))
    L1 = 0.1 * torch.abs(decoded_MLP)
    combined_sparsity = (smooth_threshold + L1).sum()
    sparsity = fixed_coeff * combined_sparsity

    mask_variance = 0.0
    mask_mean = 0.0
    for i in range(num_layers):
        mask = all_activated_nodes[:, :, i] 
        mask_variance += mask.std(dim=-1).mean()
        mask_mean += mask.mean()  
    mask_variance = mask_variance / num_layers
    mask_mean = mask_mean / num_layers

    total_loss = pred_loss + 0.4 * (-mask_variance) + sparsity + 0.001 * mask_mean
    return total_loss, pred_loss, mask_variance, mask_mean, smooth_threshold.sum()


def test_loss_fn(outputs, predictions):
    return ((outputs - predictions) ** 2).mean()


# Load autoencoder
input_size = 2
output_size = 1
num_samples = 1000
max_hidden_size = 7

num_hidden_layers = 4
input_dim = 56
hidden_dim = 768
batch_size = 1

initial_temp = 1.0
final_temp = 0.01
anneal_epochs = 3000
def get_temperature(epoch):
    return max(final_temp, initial_temp * (1 - epoch / anneal_epochs))

autoencoder = AutoEncoder(input_dim, hidden_dim, batch_size, max_hidden_size, device)

log_file = 'final.txt'
memory_log_file = 'memory.log'

checkpoint_path = f"../checkpoint/pretrain_ae.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
autoencoder.load_state_dict(checkpoint["model_state_dict"])
autoencoder.eval()
for param in autoencoder.parameters():
    param.requires_grad = False

# Load the dataset
function_dictionary = CORNN.get_benchmark_functions()
all_function_names = list(function_dictionary.keys()) 

# Train each function
for function_name in all_function_names:
    training_data, test_data = CORNN.get_scaled_function_data(function_dictionary[function_name])
    training_data[0] = training_data[0].to(device)
    training_data[1] = training_data[1].to(device)
    test_data[0] = test_data[0].to(device)
    test_data[1] = test_data[1].to(device)

    for num_layer in range (4):
      z_size = hidden_dim 
      z = torch.randn(1, max_hidden_size, hidden_dim, device=device, requires_grad=True)
      t = torch.tensor([0.0], device=device, requires_grad=True)
      node_t = torch.tensor([0.5], device=device).detach()
      out_num_hidden_layer = num_layer + 1
      # Define optimizer
      optimizer = optim.Adam([{'params': z, 'lr': 0.1}, {'params': t, 'lr': 0.001}])

      all_loss = []
      avg_loss = []
      best_loss = float('inf')
      for epoch in range (8001):
        decoder_map = {
          0: (autoencoder.decoder_7_18, autoencoder.final_layer_7_18, 14),
          1: (autoencoder.decoder_7_23, autoencoder.final_layer_7_23, 21),
          2: (autoencoder.decoder_7_31, autoencoder.final_layer_7_31, 28),
          3: (autoencoder.decoder_7_39, autoencoder.final_layer_7_39, 35),
        }
        layer_config_map = {
            1: layer_config_1HL,
            2: layer_config_2HL,
            3: layer_config_3HL,
            4: layer_config_4HL
        }

        current_decoder, current_final_layer, current_weight_col = decoder_map[num_layer]
        decoder_output = current_decoder(inputs_embeds=z).last_hidden_state 
        current_output = current_final_layer(decoder_output).view(1, autoencoder.max_hidden_size, -1)
        current_config = layer_config_map[out_num_hidden_layer]
        decoded_MLP, activation, all_activated_nodes, all_biases = parse_decoder_output(current_output, current_config, max_hidden_size, device)

        original_decoded = decoded_MLP.detach().clone()
        soft_mask = torch.sigmoid(200 * (torch.abs(decoded_MLP) - t))
        decoded_MLP = decoded_MLP * soft_mask
   
        for i in range(out_num_hidden_layer):
          start_index = i * max_hidden_size
          end_index = min((i + 1) * max_hidden_size, decoded_MLP.size(2))
          
          # Weights
          sliced_matrix = decoded_MLP[:, :, start_index:end_index]
          num_padding = max_hidden_size - 2

          # Neuron activation mask
          sliced_mask = all_activated_nodes[:, :, i]
          sliced_mask = torch.sigmoid(200 * (sliced_mask - node_t))
          sliced_mask_expanded = sliced_mask.squeeze(-1).expand(training_data[0].size(0), -1)
          sliced_mask_expanded = sliced_mask_expanded.unsqueeze(0)

          # Activation with temperature
          raw_logits = activation[i].unsqueeze(1)
          temperature = get_temperature(epoch)
          act_weights = F.softmax(raw_logits / temperature, dim=-1).unsqueeze(1)
          
          # Bias term
          bias = all_biases[i]
          bias = bias.permute(0, 2, 1)

          if i == 0:
              sliced_matrix = sliced_matrix[:, :-num_padding, :] 
              inter = torch.matmul(training_data[0], sliced_matrix) + bias
              # Activation per neuron
              sigmoid_part = torch.sigmoid(inter)
              tanh_part = torch.tanh(inter)
              leaky_relu_part = F.leaky_relu(inter, negative_slope=0.01)
              curr_result = (
                  act_weights[..., 0] * sigmoid_part +
                  act_weights[..., 1] * tanh_part +
                  act_weights[..., 2] * leaky_relu_part
              )
              curr_result = curr_result * sliced_mask_expanded  
              prev_result = curr_result
          else:
              inter = torch.matmul(prev_result, sliced_matrix) + bias
              # Activation per neuron
              sigmoid_part = torch.sigmoid(inter)
              tanh_part = torch.tanh(inter)
              leaky_relu_part = F.leaky_relu(inter, negative_slope=0.01)
              curr_result = (
                  act_weights[..., 0] * sigmoid_part +
                  act_weights[..., 1] * tanh_part +
                  act_weights[..., 2] * leaky_relu_part
              )
              curr_result = curr_result * sliced_mask_expanded  
              prev_result = curr_result

        sliced_matrix = decoded_MLP[:, :, (out_num_hidden_layer) * max_hidden_size : (out_num_hidden_layer) * max_hidden_size + 1]
        bias = all_biases[-1].permute(0, 2, 1)
        bias = bias[:, :, 0].unsqueeze(1)
        prediction = (torch.matmul(curr_result, sliced_matrix) + bias).squeeze(-1).squeeze(0)

        # Calculate loss
        loss, pred_loss, var, mean, sf = loss_fn_2(training_data[1].squeeze(-1), prediction, decoded_MLP, all_biases, t, out_num_hidden_layer)

        if epoch > (anneal_epochs) and loss < best_loss:
            best_loss = loss
            best_pred_loss = pred_loss
            best_model = torch.where(torch.abs(original_decoded) < t, torch.tensor(0.0, device=device), original_decoded)
            best_mask = all_activated_nodes.detach().clone()
            best_activation = [a.detach().clone() for a in activation]
            best_bias = [b.detach().clone() for b in all_biases]
            best_node_t = node_t
            best_t = t.detach().clone()
            best_original = [ori.detach().clone() for ori in original_decoded]


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            t.clamp_(min=0.0, max=3.0)

        all_loss.append(loss.item())
        avg_loss.append(sum(all_loss) / len(all_loss))

        if epoch % 200 == 0:
            print(f"Training -- {function_name} with {out_num_hidden_layer} HL - [Epoch {epoch}] Total Loss: {loss.item():.5f}, pred: {pred_loss.item():.5f}, t: {t.item():.4f}, mean:{mean.item():.4f}, var: {var.item():.4f}")
            # print(f"Training -- {function_name} with {out_num_hidden_layer} HL -- [Epoch {epoch}] Total Loss (pred + sparsity): {loss.item()}, t: {node_t_list}")
      
      # Record the final MLP
      with torch.no_grad():
        best_model_np = best_model.detach().cpu().numpy()
        num_rows, num_cols = best_model_np.shape[-2:] 
        best_model_np[..., -5:, :7] = 0
        best_model_np[..., :, -6:] = 0
        best_mask_np = best_mask.detach().cpu().numpy()
        num_layers = out_num_hidden_layer
        for layer in range(num_layers):
            for node_idx in range(max_hidden_size):
                if best_mask_np[0, node_idx, layer] < node_t:
                    col_index = layer * max_hidden_size + node_idx
                    best_model_np[..., :, col_index] = 0
                    if layer < num_layers:
                        row_start = (layer + 1) * max_hidden_size
                        row_end = row_start + max_hidden_size
                        best_model_np[..., node_idx, row_start:row_end] = 0

        modified_best_model = torch.tensor(best_model_np, device=device)
        with open("trained_MLP.txt", 'a') as f:
          f.write(f'Best for {function_name} with {out_num_hidden_layer} HL: {modified_best_model}, Total Loss (pred + sparsity): {best_loss.item():.4f}, best activation: {best_activation}, best bias: {best_bias}, best neuron mask: {best_mask}, best pred loss: {best_pred_loss.item():.4f}, best t: {best_t.item():.4f}, best original: {best_original}\n')





      # Test on test dataset
      total_active_nodes = 0
      for i in range(out_num_hidden_layer):
        start_index = i * max_hidden_size
        end_index = min((i + 1) * max_hidden_size, best_model.size(2))

        sliced_matrix = best_model[:, :, start_index:end_index]
        num_padding = max_hidden_size - 2

        sliced_mask = best_mask[:, :, i]
        sliced_mask = (sliced_mask >= best_node_t).int()
        active_count = sliced_mask.sum().item()
        total_active_nodes += active_count
        active_count = sliced_mask.sum().item()
        # Use hard thresholding to decide if the neuron is active or not
        sliced_mask_expanded = sliced_mask.squeeze(-1).expand(test_data[0].size(0), -1)
        sliced_mask_expanded = sliced_mask_expanded.unsqueeze(0)

        raw_logits = best_activation[i]
        act_weights = F.softmax(raw_logits, dim=-1).unsqueeze(1)

        bias = best_bias[i]
        bias = bias.permute(0, 2, 1)

        if i == 0:
            sliced_matrix = sliced_matrix[:, :-num_padding, :]
            inter = torch.matmul(test_data[0], sliced_matrix) + bias
            
            # Activation per node
            sigmoid_part = torch.sigmoid(inter)
            tanh_part = torch.tanh(inter)
            leaky_relu_part = F.leaky_relu(inter, negative_slope=0.01)
            # Use one of the activation -- hard thresholding
            act_indices = torch.argmax(act_weights.squeeze(1), dim=-1)
            curr_result = torch.zeros_like(sigmoid_part)

            for idx in range(3):
              mask = (act_indices == idx).unsqueeze(1) 
              if idx == 0:
                  curr_result += mask * sigmoid_part
              elif idx == 1:
                  curr_result += mask * tanh_part
              elif idx == 2:
                  curr_result += mask * leaky_relu_part
            curr_result = curr_result * sliced_mask_expanded  
            prev_result = curr_result

        else:
            inter = torch.matmul(prev_result, sliced_matrix) + bias 
            sigmoid_part = torch.sigmoid(inter)
            tanh_part = torch.tanh(inter)
            leaky_relu_part = F.leaky_relu(inter, negative_slope=0.01)

            act_indices = torch.argmax(act_weights.squeeze(1), dim=-1)
            curr_result = torch.zeros_like(sigmoid_part)

            for idx in range(3):
                mask = (act_indices == idx).unsqueeze(1)
                if idx == 0:
                    curr_result += mask * sigmoid_part
                elif idx == 1:
                    curr_result += mask * tanh_part
                elif idx == 2:
                    curr_result += mask * leaky_relu_part
            curr_result = curr_result * sliced_mask_expanded  
            prev_result = curr_result

      sliced_matrix = best_model[:, :, (out_num_hidden_layer) * max_hidden_size : (out_num_hidden_layer) * max_hidden_size + 1]
      bias = best_bias[-1].permute(0, 2, 1)
      bias = bias[:, :, 0].unsqueeze(1)
      prediction = (torch.matmul(curr_result, sliced_matrix)+bias).squeeze(-1).squeeze(0)

      loss = test_loss_fn(test_data[1].squeeze(-1), prediction)
      non_zero_count = torch.count_nonzero(modified_best_model).item()
      for layer, bias in enumerate(best_bias):
        if layer == len(best_bias) - 1:
            non_zero_count += torch.count_nonzero(bias[:, 0, :]).item()
        else:
            mask = best_mask[:, :, layer] 
            mask = (mask >= best_node_t).int().squeeze(0).squeeze(-1) 
            bias_values = bias.squeeze(0) 
            active_bias = bias_values[mask.bool()]
            non_zero_count += torch.count_nonzero(active_bias).item()

      with open("test_MLP.txt", 'a') as f:
        f.write(f'Test loss for {function_name} with {out_num_hidden_layer} HL -- MSE Loss: {loss.item():.4f}, Activated Nodes: {total_active_nodes}, Non-zero counts: {non_zero_count}\n')


