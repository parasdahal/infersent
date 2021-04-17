import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


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
    x = self.projection(x)
    _, (h, _) = self.lstm(x)
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

    # h0 = torch.zeros(self.num_layers * 2, self.batch_size,
    #                  self.hidden_dim).to(self.device)
    # c0 = torch.zeros(self.num_layers * 2, self.batch_size,
    #                  self.hidden_dim).to(self.device)

    x = self.projection(x)
    output, _ = self.lstm(x)

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
## InferSent model
###############################################################################
class InferSentClassifier(nn.Module):

  def __init__(self, input_dim, hidden_dim, out_dim):

    super(InferSentClassifier, self).__init__()
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
## NLI pytorch lightning model to bring it all together
###############################################################################
class NLINet(pl.LightningModule):

  def __init__(self, encoder_type, enc_hidden_dim, cls_hidden_dim, lr):
    super().__init__()
    self.save_hyperparameters()

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

    self.classifier = InferSentClassifier(self.cls_input_dim,
                                          cls_hidden_dim,
                                          out_dim=3)
    self.criterion = nn.CrossEntropyLoss()

  def forward(self, premise, hypothesis):
    premise_encoded = self.encoder(premise)
    hypothesis_encoded = self.encoder(hypothesis)
    out = self.classifier(premise_encoded, hypothesis_encoded)
    return out

  def configure_optimizers(self):
    optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams['lr'])
    return optimizer

  def training_step(self, train_batch, batch_idx):
    (premise, hypothesis), label = train_batch
    out = self.forward(premise, hypothesis)
    loss = self.criterion(out.float(), label)
    self.log("train_loss", loss)
    return loss

  def validation_step(self, val_batch, batch_idx):
    (premise, hypothesis), label = val_batch
    out = self.forward(premise, hypothesis)
    loss = self.criterion(out.float(), label)
    self.log("val_loss", loss)
    return loss

  def test_step(self, test_batch, batch_idx):
    (premise, hypothesis), label = test_batch
    out = self.forward(premise, hypothesis)
    loss = self.criterion(out.float(), label)
    self.log("test_loss", loss)
    return loss