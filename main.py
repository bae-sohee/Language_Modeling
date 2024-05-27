import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, SubsetRandomSampler

from dataset import Shakespeare
from model import CharRNN, CharLSTM

def train(model, trn_loader, device, criterion, optimizer, is_lstm=False):
    """ Train function
    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim
    Returns:
        trn_loss: average loss value
    """
    model.train()
    total_loss = 0
    for inputs, targets in trn_loader:
        inputs, targets = inputs.to(device).long(), targets.to(device).long()
        optimizer.zero_grad()

        hidden = model.init_hidden(inputs.size(0))
        if is_lstm:
            hidden = (hidden[0].to(device), hidden[1].to(device))
        else:
            hidden = hidden.to(device)       

        outputs, hidden = model(inputs, hidden)
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    trn_loss = total_loss / len(trn_loader)

    return trn_loss

def validate(model, val_loader, device, criterion, is_lstm=False):
    """ Validate function
    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function
    Returns:
        val_loss: average loss value
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device).long(), targets.to(device).long()

            hidden = model.init_hidden(inputs.size(0))
            if is_lstm:
                hidden = (hidden[0].to(device), hidden[1].to(device))
            else:
                hidden = hidden.to(device)
                
            outputs, hidden = model(inputs, hidden)
            outputs = outputs.view(-1, outputs.size(-1))  
            targets = targets.view(-1) 
            loss = criterion(outputs, targets)
            total_loss += loss.item()
        val_loss = total_loss / len(val_loader)
    return val_loss


def main():
    """ Main function
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss
    """
    parser = argparse.ArgumentParser(description='Train RNN or LSTM model.')
    parser.add_argument('--lstm', action='store_true', help='Use the LSTM model for training')
    parser.add_argument('--rnn', action='store_true', help='Use the RNN model for training')
    args = parser.parse_args()

    input_file = 'shakespeare.txt'
    dataset = Shakespeare(input_file)

    batch_size = 64
    validation_split = .2
    shuffle_dataset = True
    random_seed = 42

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(dataset.chars)
    embedding_size = 128
    hidden_size = 256
    num_layers = 3

    if args.lstm:
        model = CharLSTM(vocab_size, embedding_size, hidden_size, vocab_size, num_layers).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.0001)
        is_lstm = True
        best_model_path = 'best_char_lstm.pth'
        plot_name = 'Training and Validation Loss of LSTM'
        plot_filename = 'lstm_loss_plot.png'
    elif args.rnn:
        model = CharRNN(vocab_size, embedding_size, hidden_size, vocab_size, num_layers).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.0001)
        is_lstm = False
        best_model_path = 'best_char_rnn.pth'
        plot_name = 'Training and Validation Loss of RNN'
        plot_filename = 'rnn_loss_plot.png'
    else:
        raise ValueError("You must specify either --lstm or --rnn")

    # cost function
    criterion = nn.CrossEntropyLoss()

    num_epochs = 200
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in tqdm(range(num_epochs)):
        trn_loss = train(model, train_loader, device, criterion, optimizer, is_lstm)
        val_loss = validate(model, validation_loader, device, criterion, is_lstm)
        train_losses.append(trn_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

        print(f'Epoch [{epoch+1}/{num_epochs}]\n'
              f'Train Loss: {trn_loss:.4f}, Val Loss: {val_loss:.4f}')
        
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.scatter(best_epoch, best_val_loss, color='red', label='Best Model')
    plt.legend()
    plt.title(plot_name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(plot_filename) # Save the plot to a file
    plt.show()

if __name__ == '__main__':
    main()