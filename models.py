import torch
import torch.nn as nn


###############################################################################
## Word embedding averaging model
###############################################################################
class MeanEmbedding(nn.Module):

  def __init__(self,):

    super(MeanEmbedding, self).__init__()
    self.emb_dim = 300
    self.embedding = nn.Embedding.from_pretrained(
        torch.load('./.vector_cache/snli_vectors.pt'))

  def forward(self, x):
    embed = torch.mean(self.embedding(x), dim=1)
    return embed


###############################################################################
## LSTM Encoder model
###############################################################################
class LSTMEncoder(nn.Module):

  def __init__(
      self,
      hidden_dim,
      batch_size=64,
  ):

    super(LSTMEncoder, self).__init__()

    self.batch_size = batch_size
    self.hidden_dim = hidden_dim
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.emb_dim = 300
    self.embedding = nn.Embedding.from_pretrained(
        torch.load('./.vector_cache/snli_vectors.pt'))
    self.linear = nn.Linear(self.emb_dim, self.hidden_dim)
    self.relu = nn.ReLU()
    self.projection = nn.Sequential(self.embedding, self.linear, self.relu)

    self.lstm = nn.LSTM(input_size=self.hidden_dim,
                        hidden_size=self.hidden_dim,
                        num_layers=1,
                        bidirectional=False,
                        batch_first=True)

  def forward(self, x):
    lengths = [len(sent) for sent in x]
    x = self.projection(x)

    h0 = torch.zeros(1, x.shape[0], self.hidden_dim).to(self.device)
    c0 = torch.zeros(1, x.shape[0], self.hidden_dim).to(self.device)

    x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
    _, (h, _) = self.lstm(x, (h0, c0))
    return torch.squeeze(h)  # batch_size x hidden_dim


###############################################################################
## Bidirectional LSTM Encoder model
###############################################################################
class BiLSTMEncoder(nn.Module):

  def __init__(self, hidden_dim, maxpool=False, batch_size=64):

    super(BiLSTMEncoder, self).__init__()

    self.batch_size = batch_size
    self.hidden_dim = hidden_dim
    self.maxpool = maxpool
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.emb_dim = 300
    self.embedding = nn.Embedding.from_pretrained(
        torch.load('./.vector_cache/snli_vectors.pt'))
    self.linear = nn.Linear(self.emb_dim, self.hidden_dim)
    self.relu = nn.ReLU()
    self.projection = nn.Sequential(self.embedding, self.linear, self.relu)

    self.lstm = nn.LSTM(input_size=self.hidden_dim,
                        hidden_size=self.hidden_dim,
                        bidirectional=True,
                        batch_first=True)

  def forward(self, x):

    lengths = [len(sent) for sent in x]
    x = self.projection(x)

    h0 = torch.zeros(2, x.shape[0], self.hidden_dim).to(self.device)
    c0 = torch.zeros(2, x.shape[0], self.hidden_dim).to(self.device)

    x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
    padded_output, _ = self.lstm(x, (h0, c0))
    output, _ = nn.utils.rnn.pad_packed_sequence(padded_output,
                                                 batch_first=True)

    if self.maxpool:
      # Perform maxpool on each output time step
      max_vecs = [torch.max(x, 0)[0] for x in output]
      embed = torch.stack(max_vecs, 0)
    else:
      # Concatenate final hidden states of forward and backward directions
      embed = torch.cat(
          (output[:, -1, :self.hidden_dim], output[:, -1, self.hidden_dim:]),
          dim=1)
    return embed


###############################################################################
## Classifier MLP model
###############################################################################
class Classifier(nn.Module):

  def __init__(self, input_dim, hidden_dim, out_dim):

    super(Classifier, self).__init__()
    self.lin1 = nn.Linear(input_dim, hidden_dim)
    self.lin2 = nn.Linear(hidden_dim, hidden_dim)
    self.lin3 = nn.Linear(hidden_dim, out_dim)
    self.relu = nn.ReLU()

    self.net = nn.Sequential(self.lin1, self.relu, self.lin2, self.relu,
                             self.lin3)

  def forward(self, premise, hypothesis):
    combined = torch.cat((premise, hypothesis, torch.abs(premise - hypothesis),
                          premise * hypothesis), 1)
    out = self.net(combined)
    return out


###############################################################################
## InferSent model to bring it all together
###############################################################################


class InferSent(nn.Module):

  def __init__(self, encoder_type, enc_hidden_dim, cls_hidden_dim):
    super(InferSent, self).__init__()
    # Architecture
    if encoder_type == 'LSTM':
      self.encoder = LSTMEncoder(enc_hidden_dim)
      # Scale by 1 (direction) * 4 (concatenation of u, v |u-v| and u*v).
      self.cls_input_dim = enc_hidden_dim * 1 * 4
    elif encoder_type == 'BiLSTM':
      self.encoder = BiLSTMEncoder(enc_hidden_dim)
      self.cls_input_dim = enc_hidden_dim * 2 * 4
    elif encoder_type == 'BiLSTM-maxpool':
      self.encoder = BiLSTMEncoder(enc_hidden_dim, maxpool=True)
      self.cls_input_dim = enc_hidden_dim * 2 * 4
    elif encoder_type == 'MeanEmbedding':
      self.encoder = MeanEmbedding()
      self.cls_input_dim = 300 * 4
    else:
      raise Exception('Unknown model type')

    self.classifier = Classifier(self.cls_input_dim, cls_hidden_dim, out_dim=3)

  def forward(self, batch):
    (premise, hypothesis), _ = batch
    premise_encoded = self.encoder(premise)
    hypothesis_encoded = self.encoder(hypothesis)
    out = self.classifier(premise_encoded, hypothesis_encoded)
    return out

  def encode(self, sent):
    return self.encoder(sent)
