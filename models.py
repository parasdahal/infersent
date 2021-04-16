import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):

  def __init__(
      self,
      seq_length,
      input_dim,
      hidden_dim,
      device,
      num_layers=1,
      batch_size=64,
  ):

    super(LSTMEncoder, self).__init__()

    self.batch_size = batch_size
    self.seq_length = seq_length
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.device = device

    self.emb_dim = 300
    self.embedding = nn.Embedding.from_pretrained(
        torch.load('./logs/snli_glove.pt'))
    self.linear = nn.Linear(self.emb_dim, self.hidden_dim)
    self.relu = nn.ReLU()
    self.projection = nn.Sequential(self.embedding, self.linear, self.relu)

    self.lstm = nn.LSTM(self.hidden_dim,
                        self.hidden_dim,
                        self.num_layers,
                        bidirectional=False,
                        batch_first=True)

  def forward(self, x):
    x = self.projection(x)
    _, (h, _) = self.lstm(x, (h0, c0))

    return h


class BiLSTMEncoder(nn.Module):

  def __init__(self,
               seq_length,
               input_dim,
               hidden_dim,
               device,
               maxpool=False,
               num_layers=1,
               batch_size=64):

    super(BiLSTMEncoder, self).__init__()

    self.batch_size = batch_size
    self.seq_length = seq_length
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.device = device

    self.emb_dim = 300
    self.embedding = nn.Embedding(self.seq_length, self.emb_dim)
    self.linear = nn.Linear(self.emb_dim, self.hidden_dim)
    self.relu = nn.ReLU()
    self.projection = nn.Sequential(self.embedding, self.linear, self.relu)

    self.lstm = nn.LSTM(self.hidden_dim,
                        self.hidden_dim,
                        self.num_layers,
                        bidirectional=True,
                        batch_first=True)

  def forward(self, x):

    # h0 = torch.zeros(self.num_layers * 2, self.batch_size,
    #                  self.hidden_dim).to(self.device)
    # c0 = torch.zeros(self.num_layers * 2, self.batch_size,
    #                  self.hidden_dim).to(self.device)

    x = self.projection(x)
    output, (h_n, c_n) = self.lstm(x)

    if maxpool:
      # Perform maxpool on each output time step
      embed = torch.stack([torch.max(x, 0) for x in output], 0)
    else:
      # Concatenate final hidden states of forward and backward hidden states
      embed = torch.cat(
          (output[:, :, :self.hidden_dim], output[:, :, self.hidden_dim:]),
          dim=1)
    return embed


class SentClassifier(nn.Module):

  def __init__(self, input_dim, hidden_dim, out_dim, batch_size, device):

    super(SentClassifier, self).__init__()
    self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
    self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
    self.lin3 = nn.Linear(self.dim, self.out_dim)
    self.relu = nn.ReLU()

    self.net = nn.Sequential(self.lin1, self.relu, self.lin2, self.relu,
                             self.lin3)

  def forward(self, premise, hypothesis):
    combined = torch.cat((premise, hypothesis, torch.abs(premise - hypothesis),
                          premise * hypothesis), 1)
    out = self.net(combined)
    return out