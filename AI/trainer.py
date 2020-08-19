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

def run_epoch(data, model, optimizer, device):
    ''' One model pass of data. Returns loss & Accuracy '''
    losses = []
    accuracies = []
    loss_function = nn.MSELoss()
    hidden_Re, hidden_Im = model.init_hidden()

    # Use optimizer when training
    is_training = model.training

    # Iterate through batches
    for batch in tqdm(data):
        # Detach the hidden state
        hidden_Re, hidden_Im = repackage_hidden(hidden_Re), repackage_hidden(hidden_Im)

        # Grab x and y
        x, target = batch['x'], batch['y']

        # Get output predictions
        out, hidden_Re, hidden_Im = model(x, hidden_Re, hidden_Im)

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