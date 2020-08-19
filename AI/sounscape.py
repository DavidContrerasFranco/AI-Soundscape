import numpy as np
import torch
import torch.nn as nn
from utils import batchify_data
from trainer import train_model
import time
import sys
from scipy.io import loadmat
from math import ceil
sys.path.append("..")


## Model specification
class Soundscape(nn.Module):

    def __init__(self, input_dimension, seq_len=60, n_layers=1, batch_size=1, input_size_reducer=1):
        super(Soundscape, self).__init__()

        # General parameters
        self.input_dimension = input_dimension
        self.hidden_dim = ceil(input_dimension * input_size_reducer)
        self.n_layers = n_layers

        self.linearHidden = nn.Linear(input_dimension * 2, self.hidden_dim)
        self.lstm_Re = nn.LSTM(self.hidden_dim, self.hidden_dim, n_layers)
        self.lstm_Im = nn.LSTM(self.hidden_dim, self.hidden_dim, n_layers)
        self.out = nn.Linear(self.hidden_dim * 2, input_dimension*2)

    def forward(self, x, hidden_Re, hidden_Im):
        # Split Real & Imaginary parts
        x_reshaped = torch.reshape(torch.transpose(x, 0, 1), (1, 60, self.input_dimension*2))

        x_hidden = self.linearHidden(x_reshaped)
        
        lstm_Re, hidden_Re = self.lstm_Re(x_hidden, hidden_Re)
        lstm_Im, hidden_Re = self.lstm_Im(x_hidden, hidden_Im)

        out = self.out(torch.cat((lstm_Re, lstm_Im), dim=2))

        # Resahpe to match input
        out = torch.reshape(out, (self.input_dimension, 2, 60))

        return out, hidden_Re, hidden_Im

    def init_hidden(self, seq_len=60, batch_size=1):
        hidden_state_Re = torch.randn(batch_size, seq_len, self.hidden_dim).to(device)
        cell_state_Re = torch.randn(batch_size, seq_len, self.hidden_dim).to(device)
        hidden_state_Im = torch.randn(batch_size, seq_len, self.hidden_dim).to(device)
        cell_state_Im = torch.randn(batch_size, seq_len, self.hidden_dim).to(device)

        hidden_Re = (hidden_state_Re, cell_state_Re)
        hidden_Im = (hidden_state_Im, cell_state_Im)

        return hidden_Re, hidden_Im


def main():
    # Load the dataset
    mat_contents = loadmat('../DataProcess/fsamples.mat')
    raw_data = mat_contents['Vec']
    input_dimension = raw_data.shape[0]
    
    # Batchify Data
    batch_size = 60
    batched_Data = batchify_data(raw_data, batch_size, device)

    # Split into train and dev
    dev_split_index = int(9 * np.array(batched_Data).shape[0] / 10)
    dev_batches   = batched_Data[dev_split_index:]
    train_batches = batched_Data[:dev_split_index]
    
    # Load model
    model = Soundscape(input_dimension, input_size_reducer=0.3)

    ### For GPU Use: If Enabled###
    model.to(device)
    
    train_model(train_batches, dev_batches, model, device)


if __name__ == '__main__':
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        dev = "cuda:0"
        torch.backends.cudnn.fastest = True
        torch.backends.cudnn.benchmark = True
    else:
        dev = "cpu"

    device = torch.device(dev)
    
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)
    start = time.process_time()
    main()
    print("Time taken = ", time.process_time() - start)