from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import compute_accuracy, repackage_hidden

def train_model(train_data, dev_data, model, device, lr=0.001,
                                                     betas=(0.9, 0.999),
                                                     eps=1e-08,
                                                     weight_decay=0,
                                                     amsgrad=False,
                                                     n_epochs=30):
    ''' Train a model for N epochs given data and hyper-params '''
    # Adam optimizer with default parameters
    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           betas=betas,
                           eps=eps,
                           weight_decay=weight_decay,
                           amsgrad=amsgrad)

    for epoch in range(1, n_epochs):
        print("-------------\nEpoch {}:\n".format(epoch))

        # Training
        loss, acc = run_epoch(train_data, model.train(), optimizer, device)
        print('Train loss: {:.6f} | Train accuracy: {:.6f}'.format(loss, acc))

        # Validation
        val_loss, val_acc = run_epoch(dev_data, model.eval(), optimizer, device)
        print('Validation loss:   {:.6f} | Validation accuracy:   {:.6f}'.format(val_loss, val_acc))
        # Save model
        torch.save(model, 'ai_soundscape.pt')
    return val_acc

def run_epoch(data, model, optimizer, device, seq_len=60):
    ''' One model pass of data. Returns loss & Accuracy '''
    losses = []
    accuracies = []
    loss_function = nn.MSELoss()

    # Use optimizer when training
    is_training = model.training

    if is_training:
        hidden_Re, hidden_Im = model.init_hidden_noise(seq_len=seq_len, device=device)
    else:
        hidden_Re, hidden_Im = model.init_hidden_silence(seq_len=seq_len, device=device)

    # Iterate through batches
    for batch in tqdm(data):
        # Detach the hidden state
        hidden_Re, hidden_Im = repackage_hidden(hidden_Re), repackage_hidden(hidden_Im)

        # Grab x and y
        x, target = batch['x'], batch['y']

        # Get output predictions
        out, hidden_Re, hidden_Im = model(x, hidden_Re, hidden_Im, seq_len=seq_len)

        # Predict and store accuracy
        accuracies.append(compute_accuracy(out, target, device).item())

        # Compute loss
        loss = loss_function(out, target)
        losses.append(loss.data.item())

        # If training, do an update.
        if is_training:
            # Clear gradients before each instance
            model.zero_grad()
            # Compute gradients
            loss.backward()
            # Update parameters
            optimizer.step()

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    avg_accuracy = np.mean(accuracies)
    return avg_loss, avg_accuracy

def use_model(init_data, model, hidden_Re, hidden_Im, duration, seq_len=60):
    ''' One model pass of data. Returns data generated '''

    # Raw data
    current_data = init_data
    final_raw_data = np.zeros((init_data.shape[0], init_data.shape[1], seq_len*duration))

    # Iterate through batches
    for batch in tqdm(range(duration)):
        # Detach the hidden state
        hidden_Re, hidden_Im = repackage_hidden(hidden_Re), repackage_hidden(hidden_Im)

        # Get output predictions
        current_data, hidden_Re, hidden_Im = model(current_data, hidden_Re, hidden_Im, seq_len=seq_len)

        # Store current data
        final_raw_data[:,:, batch:batch + seq_len] = current_data.cpu().detach().numpy()

    return final_raw_data