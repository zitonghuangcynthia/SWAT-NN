import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import random


NUM_HIDDEN_LAYERS = [1, 2, 3, 4]
SPARSITY_LEVELS = [0.0, 0.2, 0.4]



# dataset generation
# Generate random uniform input between -1 and 1
def generate_input(input_size, batch_size, device=None):
    num_points = 1000
    x1 = 2 * torch.rand(num_points, device=device) - 1 
    x2 = 2 * torch.rand(num_points, device=device) - 1
    inputs = torch.stack((x1, x2), dim=-1)  
    batched_inputs = inputs.unsqueeze(0).repeat(batch_size, 1, 1)
    return batched_inputs



# Generate random weights between [-5, 5], bias between [-1, 1]
def generate_weight(input_size, output_size, num_samples, hidden_size, sparsity_level=0.0, device=None):
    num_hidden_layers = np.random.choice(NUM_HIDDEN_LAYERS)

    layers = [input_size] + [hidden_size] * num_hidden_layers + [output_size]
    weights, active_masks, activations = [], [], []

    biases = []
    for i in range(len(layers) - 1):
      weight = torch.empty((num_samples, layers[i], layers[i+1]), device=device).uniform_(-5, 5)
      bias = torch.empty((num_samples, layers[i+1]), device=device).uniform_(-1, 1)
      biases.append(bias)

      mask = torch.rand_like(weight) > sparsity_level  
      weight = weight * mask 
      weights.append(weight)

      if i < len(layers) - 2:  
        active_mask = torch.randint(0, 2, (num_samples, hidden_size), device=device, dtype=torch.float32)
        active_masks.append(active_mask)
        choices = torch.randint(0, 3, (num_samples, hidden_size), device=device)
        activation_onehot = F.one_hot(choices, num_classes=3).float() 
        activations.append(activation_onehot)

    return weights, biases, num_hidden_layers, active_masks, activations



# Calculate forward pass for MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, custom_weights, num_hidden_layers, all_activate_masks, activations_func, biases):
        super(MLP, self).__init__()

        self.num_hidden_layers = num_hidden_layers
        self.all_activate_masks = all_activate_masks
        self.activations_func = activations_func

        self.fc_weights = custom_weights[:-1]
        self.output_weight = custom_weights[-1]
        self.biases = biases

    def forward(self, x):
        # print(self.biases)
        for i, weight in enumerate(self.fc_weights):
          current_mask = self.all_activate_masks[:, :, i]
          current_mask_expanded = current_mask.unsqueeze(1).expand(-1, x.shape[1], -1)
          bias = self.biases[i]

          a = F.softmax(self.activations_func[i], dim=-1) 
          z_med = torch.matmul(x, weight) + bias[:, None, :]
          act = (
              a[:, None, :, 0] * torch.sigmoid(z_med) +
              a[:, None, :, 1] * torch.tanh(z_med) +
              a[:, None, :, 2] * F.leaky_relu(z_med, negative_slope=0.01)
          )
          x = act * current_mask_expanded
        return torch.matmul(x, self.output_weight) + self.biases[-1][:, None, :]



# Process weights to match the input for autoencoder
def preprocess_weights(weights, num_samples, active_masks, activations_funcs, biases, device=None):
    max_rows = max(weight.shape[1] for weight in weights) 
    max_cols = max(weight.shape[2] for weight in weights) 
    stacked_weights = []
    stacked_masks = []

    for i, weight in enumerate(weights):
        pad_rows = max_rows - weight.shape[1]
        pad_cols = max_cols - weight.shape[2]
        padded_weight = F.pad(weight, (0, pad_cols, 0, pad_rows)) 
        padded_weight_mask = F.pad(torch.ones_like(weight, dtype=torch.float32, device=device), (0, pad_cols, 0, pad_rows), value=0)

        # Build current layer block
        layer_blocks = [padded_weight]
        layer_mask_blocks = [padded_weight_mask]
        bias = biases[i] 
        bias = bias.unsqueeze(2) 
        bias_padded = F.pad(bias, (0, 0, 0, pad_cols), value=0) 
        bias_mask = torch.ones_like(bias_padded, device=device)
        layer_blocks.append(bias_padded)
        layer_mask_blocks.append(bias_mask)
        

        # Only add activation info if not the last layer
        if i < len(weights) - 1:
            act = activations_funcs[i]
            pad_act_rows = max_rows - act.shape[1]
            act_padded = F.pad(act, (0, 0, 0, pad_act_rows)) 
            act_mask_padded = F.pad(torch.ones_like(act, device=device), (0, 0, 0, pad_act_rows), value=0)

            layer_blocks.append(act_padded)
            layer_mask_blocks.append(act_mask_padded)

        # concat this layer horizontally
        stacked_weights.append(torch.cat(layer_blocks, dim=2))
        stacked_masks.append(torch.cat(layer_mask_blocks, dim=2))

    # concat all layers horizontally
    stacked_weights = torch.cat(stacked_weights, dim=2)
    stacked_masks = torch.cat(stacked_masks, dim=2)

    # add active masks at the end
    all_active_masks = torch.cat([am.unsqueeze(2) for am in active_masks], dim=2)
    stacked_weights = torch.cat((stacked_weights, all_active_masks), dim=2)
    extra_mask = torch.ones_like(all_active_masks, device=device)
    stacked_masks = torch.cat((stacked_masks, extra_mask), dim=2)

    return stacked_weights, stacked_masks, all_active_masks





# Generate the dataset
def generate_dataset(batch_size, input_size, output_size, hidden_size, device=None):
  inputs = generate_input(input_size, batch_size, device) 
  sparsity_level = np.random.choice(SPARSITY_LEVELS)

  weights, biases, num_hidden_layers, active_masks, activation_funcs = generate_weight(input_size, output_size, batch_size, hidden_size, sparsity_level, device)
  preprocessed_weights, stacked_masks, all_activate_masks = preprocess_weights(weights, batch_size, active_masks, activation_funcs, biases, device) 

  mlp_model = MLP(input_size, hidden_size, output_size, weights, num_hidden_layers, all_activate_masks, activation_funcs, biases).to(device)
  outputs = mlp_model(inputs)

  return inputs, preprocessed_weights, outputs, stacked_masks, num_hidden_layers