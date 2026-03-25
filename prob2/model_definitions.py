import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Vanilla RNN Implementation
class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        return self.fc(output), hidden

# 2. Bidirectional LSTM (BLSTM)
class BLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        # Bidirectional processing
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        return self.fc(output), hidden

# 3. RNN with Basic Attention
class AttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttentionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.attn = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        # Basic Attention mechanism
        attn_weights = F.softmax(self.attn(output), dim=1)
        context = attn_weights * output
        return self.fc(context), hidden