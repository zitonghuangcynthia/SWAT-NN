import torch
import torch.nn as nn
from transformers import GPT2Model
import torch.nn.functional as F


def parse_decoder_output(output, config, max_hidden_size, device=None):
    """
    Given a decoder output and layer config, extract:
    - stacked_weights: full [B, H, all_weight_columns]
    - activations_list: per-layer activation function distributions (softmax)
    - structure_mask: node-wise mask
    - bias_list: all bias terms
    """
    weights_blocks = []
    activations_list = []
    bias_list = []

    for layer_idx in range(len(config)):
        if layer_idx not in config:
            continue

        # Extract weights for this layer
        w_start, w_end = config[layer_idx]["weight_cols"]
        weights = output[:, :, w_start:w_end] 
        weights_blocks.append(weights)

        # Extract bias term (should be directly after weight block)
        bias = output[:, :, w_end:w_end + 1] 
        bias_list.append(bias)

        # Extract activation logits and apply softmax
        if "act_cols" in config[layer_idx]:
            a_start, a_end = config[layer_idx]["act_cols"]
            act_logits = output[:, :, a_start:a_end]
            act_softmax = F.softmax(act_logits.view(-1, 3), dim=-1).view(-1, max_hidden_size, 3)
            activations_list.append(act_softmax)

    # Concatenate all weights in column dimension
    stacked_weights = torch.cat(weights_blocks, dim=2)  # [B, H, total_weight_cols]

    # Extract structure mask if available
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

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, max_hidden_size, device=None):
        super(AutoEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.max_hidden_size = max_hidden_size

        self.encoder_gpt2 = GPT2Model.from_pretrained("gpt2").to(device)
        self.decoder_7_18 = GPT2Model.from_pretrained("gpt2").to(device)
        self.decoder_7_23 = GPT2Model.from_pretrained("gpt2").to(device)
        self.decoder_7_31 = GPT2Model.from_pretrained("gpt2").to(device)
        self.decoder_7_39 = GPT2Model.from_pretrained("gpt2").to(device)
        
        self.encoder_linear = nn.Linear(self.input_dim*2, self.hidden_dim).to(device)  # Projection layer

        self.final_layer_7_18 = nn.Linear(self.hidden_dim, self.input_dim).to(device)
        self.final_layer_7_23 = nn.Linear(self.hidden_dim, self.input_dim).to(device)
        self.final_layer_7_31 = nn.Linear(self.hidden_dim, self.input_dim).to(device)
        self.final_layer_7_39 = nn.Linear(self.hidden_dim, self.input_dim).to(device)

        self.positional_embedding = nn.Parameter(torch.zeros(1, 7, self.hidden_dim)).to(device)
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        self._init_weights()

        self.device = device


    def _init_weights(self):
        nn.init.xavier_uniform_(self.encoder_linear.weight)
        nn.init.xavier_uniform_(self.final_layer_7_18.weight)
        nn.init.xavier_uniform_(self.final_layer_7_23.weight)
        nn.init.xavier_uniform_(self.final_layer_7_31.weight)
        nn.init.xavier_uniform_(self.final_layer_7_39.weight)


    def forward(self, x):
        batch_size, seq_len, feature_dim, channels = x.shape
        x = x.view(batch_size, seq_len, feature_dim * channels)
        x = self.encoder_linear(x)
        x = x.view(batch_size, seq_len, -1)
        x = x + self.positional_embedding[:, :seq_len, :]
        
        # Encoder forward
        encoder_output = self.encoder_gpt2(inputs_embeds=x).last_hidden_state

        # Decoder forward
        output_18 = self.final_layer_7_18(self.decoder_7_18(inputs_embeds=encoder_output).last_hidden_state)
        output_23 = self.final_layer_7_23(self.decoder_7_23(inputs_embeds=encoder_output).last_hidden_state)
        output_31 = self.final_layer_7_31(self.decoder_7_31(inputs_embeds=encoder_output).last_hidden_state)
        output_39 = self.final_layer_7_39(self.decoder_7_39(inputs_embeds=encoder_output).last_hidden_state)

        # Parse outputs
        w1, a1, m1, b1 = parse_decoder_output(output_18, layer_config_1HL, self.max_hidden_size, x.device)
        w2, a2, m2, b2 = parse_decoder_output(output_23, layer_config_2HL, self.max_hidden_size, x.device)
        w3, a3, m3, b3 = parse_decoder_output(output_31, layer_config_3HL, self.max_hidden_size, x.device)
        w4, a4, m4, b4 = parse_decoder_output(output_39, layer_config_4HL, self.max_hidden_size, x.device)

        # Combine results
        all_weights = [w1, w2, w3, w4]          # list of list of weight tensors
        all_activations = [a1, a2, a3, a4]      # list of list of activation tensors
        all_structure_masks = [m1, m2, m3, m4]  # list of mask tensors
        all_biases = [b1, b2, b3, b4]

        return all_weights, all_activations, all_structure_masks, all_biases