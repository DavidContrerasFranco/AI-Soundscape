import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def batchify_data(data, batch_size, device):
    ''' Separates samples into batches per minute '''
    N = data.shape[2]
    # Samples start with silence
    header = np.zeros((data.shape[0], data.shape[1], 1))
    batches = []
    for i in range(0, N, batch_size):
        batches.append({
            'x': torch.tensor(np.append(header, data[:,:,i:i+batch_size - 1], axis=2), dtype=torch.float32).to(device),
            'y': torch.tensor(data[:,:,i:i+batch_size], dtype=torch.float32).to(device)
        })
    return batches

def compute_accuracy(predictions, y, device):
    ''' Computes the cosien similarity of predictions '''
    cos = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
    return torch.mean(cos(predictions, y))

def repackage_hidden(hidden):
    ''' Wraps hidden states in new Variables, to detach them from their history. '''
    hidden = tuple([e.data for e in hidden])
    return hidden