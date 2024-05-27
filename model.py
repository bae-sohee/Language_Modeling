import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, batch_first=True)  
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output.reshape(output.size(0) * output.size(1), output.size(2)))
        return output, hidden

    def init_hidden(self, batch_size):
        initial_hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return initial_hidden


class CharLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers=1):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedded, hidden)
        output = self.fc(output.reshape(output.size(0) * output.size(1), output.size(2)))
        return output, (hidden, cell)
    
    def init_hidden(self, batch_size):
        initial_hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size), 
                          torch.zeros(self.num_layers, batch_size, self.hidden_size))
        return initial_hidden