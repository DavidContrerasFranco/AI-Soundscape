import numpy as np
import torch
import torch.nn as nn
from utils import batchify_data
from trainer import train_model
import time
import sys
from scipy.io import loadmat
sys.path.append("..")


## Model specification
class Soundscape(nn.Module):

    def __init__(self, input_dimension):
        super(Soundscape, self).__init__()
        self.input_dimension = input_dimension
        self.linearHidden = nn.Linear(input_dimension * 2, input_dimension * 2)
        self.lstm_1 = nn.LSTM(input_dimension * 2, input_dimension)
        self.lstm_2 = nn.LSTM(input_dimension * 2, input_dimension)
        self.outRe = nn.Linear(input_dimension, input_dimension)
        self.outIm = nn.Linear(input_dimension, input_dimension)

    def forward(self, x):
        # Split Real & Imaginary parts
        x_reshaped = torch.reshape(torch.transpose(x, 0, 1), (1, self.input_dimension*2)).squeeze()

        x_hidden = self.linearHidden(x_reshaped)
        
        # LSTM input is a 3D Tenso: (the sequence itself, instances in the mini-batch = 1, elements of the input)
        lstm1_re = self.lstm_1(x_hidden)
        lstm1_im = self.lstm_2(x_hidden)

        out_re = self.outRe(lstm1_re)
        out_im = self.outIm(lstm1_im)

        # Join Real & Imaginary parts
        out = torch.transpose(torch.reshape(torch.cat((out_re, out_im), 0), (2, self.input_dimension)), 0, 1)

        return out

def main():
    # Load the dataset
    mat_contents = loadmat('fsamples.mat')
    raw_data = mat_contents['Vec']
    input_dimension = raw_data.shape[0]
    
    # Batchify Data
    batch_size = 60
    batched_Data = batchify_data(raw_data, batch_size)

    # Split into train and dev
    dev_split_index = int(9 * np.array(batched_Data).shape[0] / 10)
    dev_batches   = batched_Data[dev_split_index:]
    train_batches = batched_Data[:dev_split_index]
    
    # Load model
    model = Soundscape(input_dimension)

    ### For GPU Use ###
    model.cuda()

    train_model(train_batches, dev_batches, model)

    # ## Evaluate the model on test data
    # loss, accuracy = run_epoch(test_batches, model.eval(), None)

    # print ("Loss on test set:"  + str(loss) + " Accuracy on test set: " + str(accuracy))


if __name__ == '__main__':
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