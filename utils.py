import argparse


def parse_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  # Model hyperparameters
  parser.add_argument(
      '--encoder_type',
      default='BiLSTM',
      type=str,
      help='What encoder model to use',
      choices=['MeanEmbedding', 'LSTM', 'BiLSTM', 'BiLSTM-maxpool'])
  parser.add_argument('--enc_hidden_dim',
                      default=1028,
                      type=int,
                      help='Size of hidden layers of encoder models')
  parser.add_argument('--cls_hidden_dim',
                      default=512,
                      type=int,
                      help='Size of hidden layers of classifier model')
  # Optimizer hyperparameters
  parser.add_argument('--lr',
                      default=0.001,
                      type=float,
                      help='Learning rate to use')
  parser.add_argument('--batch_size',
                      default=128,
                      type=int,
                      help='Minibatch size')
  # Other hyperparameters
  parser.add_argument('--epochs',
                      default=5,
                      type=int,
                      help='Max number of epochs')
  parser.add_argument('--seed', default=42, type=int, help='Random seed')
  parser.add_argument('--limit_train_batches',
                      default=1.0,
                      type=float,
                      help='Percentage of data to use for training')
  parser.add_argument('--refresh_rate',
                      default=10,
                      type=int,
                      help='Progress bar refresh rate')
  parser.add_argument('--log_dir', default='./logs', type=str)
  return parser.parse_args()