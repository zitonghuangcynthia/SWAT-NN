import torch
import torch.nn as nn
from transformers import GPT2Model



class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, max_hidden_size, device):
        super(AutoEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.max_hidden_size = max_hidden_size

        # GPT-2 based encoder and decoder (4-to-2 autoencoder)
        self.encoder_gpt2 = GPT2Model.from_pretrained("gpt2").to(device)
        self.decoder_7_23 = GPT2Model.from_pretrained("gpt2").to(device)
        self.final_layer_7_23 = nn.Linear(self.hidden_dim, self.input_dim).to(device)
        
        self.encoder_linear = nn.Linear(self.input_dim*2, self.hidden_dim).to(device)
        self.final_layer_7_23 = nn.Linear(self.hidden_dim, self.input_dim).to(device)
        self._init_weights()
        self.device = device


    def _init_weights(self):
        nn.init.xavier_uniform_(self.encoder_linear.weight)
        nn.init.xavier_uniform_(self.final_layer_7_23.weight)
    
    def process_mask(self, x, num_mask_columns, start_col=0):
        mask = torch.sigmoid(x[:, :, start_col:start_col + num_mask_columns]) 
        return mask


    def forward(self, x):

        batch_size, seq_len, feature_dim, channels = x.shape
        x = x.view(batch_size, seq_len, feature_dim * channels)
        x = self.encoder_linear(x)
        x = x.view(batch_size, seq_len, -1)
        encoder_output = self.encoder_gpt2(inputs_embeds=x).last_hidden_state  # GPT-2 Encoder output
        output_7_23 = self.decoder_7_23(inputs_embeds=encoder_output).last_hidden_state
        output_23 = self.final_layer_7_23(output_7_23).view(batch_size, self.max_hidden_size, -1)
        reconstructed_weights_23 = output_23[:, :, :21]

        return reconstructed_weights_23