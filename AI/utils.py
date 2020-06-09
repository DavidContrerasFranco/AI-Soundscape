import numpy as np
import torch
import torch.nn as nn

def batchify_data(data, batch_size):
    ''' Separates samples into batches per minute '''
    N = data.shape[2]
    # Samples start with silence
    header = np.zeros((data.shape[0], data.shape[1], 1))
    batches = []
    for i in range(0, N, batch_size):
        batches.append({
            'x': torch.tensor(np.append(header, data[:,:,i:i+batch_size - 1], axis=2), dtype=torch.float32).cuda(),
            'y': torch.tensor(data[:,:,i:i+batch_size], dtype=torch.float32).cuda()
        })
    return batches

def compute_accuracy(predictions, y):
    ''' Computes the cosien similarity of predictions '''
    cos = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
    return torch.mean(cos(predictions, y))