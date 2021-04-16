import torch
import torch.nn as nn


class biLSTM(nn.Module):

  def __init__(self, seq_length, input_dim, hidden_dim, num_layers, batch_size,
               device):

    super(biLSTM, self).__init__()

    self.batch_size = batch_size
    self.seq_length = seq_length
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.device = device

    self.emb_dim = 300
    self.embedding = nn.Embedding(seq_length, self.emb_dim)
    self.linear = nn.Linear(self.emb_dim, self.hidden_dim)
    self.relu = nn.ReLU()
    self.projection = nn.Sequential(self.embedding, self.linear, self.relu)

    self.lstm_ltr = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.num_layers,
                            batch_first=True)
    self.lstm_rtl = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.num_layers,
                            batch_first=True)

  def forward(self, x):

    initial_h = torch.zeros(self.batch_size, self.hidden_dim).to(self.device)
    initial_c = torch.zeros(self.batch_size, self.hidden_dim).to(self.device)

    x = self.projection(x)

    _, _, h_ltr = self.lstm_ltr(x, initial_c, initial_h)
    _, _, h_rtl = self.lstm_rtl(torch.flip(x, [1]), initial_c, initial_h)

    # Concatenate hidden states
    embed = torch.cat((h_ltr, h_rtl), dim=1)
    return embed