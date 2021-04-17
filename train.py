import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data import SNLIData
from models import NLINet


def train(args):

  pl.seed_everything(args.seed)

  os.makedirs(args.log_dir, exist_ok=True)

  train_loader, val_loader, test_loader = SNLIData(
      batch_size=args.batch_size).get_iters()

  trainer = pl.Trainer(default_root_dir=args.log_dir,
                       checkpoint_callback=ModelCheckpoint(
                           save_weights_only=True,
                           mode="min",
                           monitor="val_loss"),
                       logger=TensorBoardLogger(args.log_dir),
                       gpus=1 if torch.cuda.is_available() else 0,
                       max_epochs=args.epochs,
                       progress_bar_refresh_rate=1)

  model = NLINet(encoder_type=args.encoder_type,
                 enc_hidden_dim=128,
                 cls_hidden_dim=512,
                 lr=args.lr)

  # Training
  trainer.fit(model, train_loader, val_loader)

  # Testing
  model = NLINet.load_from_checkpoint(
      trainer.checkpoint_callback.best_model_path)
  test_result = trainer.test(model, test_dataloaders=test_loader, verbose=True)
  return test_result


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # Model hyperparameters
  parser.add_argument(
      '--encoder_type',
      default='BiLSTM',
      type=str,
      help='What encoder model to use',
      choices=['MeanEmbedding', 'LSTM', 'BiLSTM', 'BiLSTM-maxpool'])
  parser.add_argument('--hidden_dim',
                      default=256,
                      type=int,
                      help='Size of hidden layers')
  # Optimizer hyperparameters
  parser.add_argument('--lr',
                      default=0.001,
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
  parser.add_argument('--seed', default=42, type=int, help='Random seed')
  parser.add_argument('--log_dir', default='./logs', type=str)
  args = parser.parse_args()

  train(args)