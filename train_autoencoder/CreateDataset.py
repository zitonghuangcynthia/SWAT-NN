from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import random
from data import generate_dataset



def CreateDataset(input_size, output_size, max_hidden_size, batch_size, device=None):

  inputs, weights, outputs, masks, num_hidden_layers = generate_dataset(batch_size, input_size, output_size, max_hidden_size, device)

  return inputs, weights, outputs, masks, num_hidden_layers