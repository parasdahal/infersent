import os
import argparse
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data import SNLIData
from models import LSTMEncoder, BiLSTMEncoder, SentClassifier

class NLI(pl.LightningModule):

  def __init__(self, model_name, hidden_dims, num_filters, lr):
    super().__init__()
    self.save_hyperparameters()
    if model_name == 'LSTM':
      self.encoder = LSTMEncoder()
    elif model_name == 'BiLSTM'
      self.encoder = BiLSTMEncoder()
    elif model_name == 'BiLSTM-maxpool'
      self.encoder = BiLSTMEncoder(maxpool=True)
    self.classifier = SentClassifier()

  def forward(self, premise, hypothesis):
    premise_encoded = self.encoder(premise)
    hypothesis_encoded = self.encoder(hypothesis)
    out = self.classifier(premise_encoded, hypothesis_encoded)
    return out

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    return optimizer

  def training_step(self, train_batch, batch_idx):
    (premise, hypothesis), label = train_batch
    out = self.forward(premise, hypothesis)
    loss = F.cross_entropy(label, out)
    self.log("train_loss", loss)
    return loss

  def validation_step(self, val_batch, batch_idx):
    (premise, hypothesis), label = val_batch
    out = self.forward(premise, hypothesis)
    loss = F.cross_entropy(label, out)
    self.log("val_loss", loss)
    return loss

  def test_step(self, test_batch, batch_idx):
    (premise, hypothesis), label = test_batch
    out = self.forward(premise, hypothesis)
    loss = F.cross_entropy(label, out)
    self.log("test_loss", loss)
    return loss

def train(args):
  os.makedirs(args.log_dir, exist_ok=True)
  train_loader, val_loader, test_loader = SNLIData(batch_size=args.batch_size)

  # Create a PyTorch Lightning trainer with the generation callback
  gen_callback = GenerateCallback(save_to_disk=True)
  trainer = pl.Trainer(default_root_dir=args.log_dir,
                       checkpoint_callback=ModelCheckpoint(
                        save_weights_only=True,
                        mode="min",
                        monitor="val_loss"),
                       gpus=1 if torch.cuda.is_available() else 0,
                       max_epochs=args.epochs,
                       progress_bar_refresh_rate=1)

  # Create model
  pl.seed_everything(args.seed)  # To be reproducible
  model = NLI(model_name=args.model,
              hidden_dims=args.hidden_dims,
              lr=args.lr)

  # Training
  trainer.fit(model, train_loader, val_loader)

  # Testing
  model = NLI.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
  test_result = trainer.test(model, test_dataloaders=test_loader, verbose=True)
  return test_result


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # Model hyperparameters
  parser.add_argument('--model',
                      default='BiLSTM',
                      type=str,
                      help='What model to use',
                      choices=['Base', 'LSTM', 'BiLSTM', 'BiLSTM-maxpool'])
  parser.add_argument('--hidden_dim',
                      default=512,
                      type=int,
                      help='Size of hidden layers')
  # Optimizer hyperparameters
  parser.add_argument('--lr',
                      default=1e-3,
                      type=float,
                      help='Learning rate to use')
  parser.add_argument('--batch_size',
                      default=64,
                      type=int,
                      help='Minibatch size')

  # Other hyperparameters
  parser.add_argument('--epochs',
                      default=10,
                      type=int,
                      help='Max number of epochs')
  parser.add_argument('--seed',
                      default=42,
                      type=int,
                      help='Random seed')
  args = parser.parse_args()

  train(args)
