import os
from utils import parse_args
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from data import SNLIData
from models import InferSent


class NLINet(pl.LightningModule):

  def __init__(self, encoder_type, enc_hidden_dim, cls_hidden_dim, lr,
               dataset_sizes):
    super().__init__()
    self.save_hyperparameters()
    self.model = InferSent(encoder_type, enc_hidden_dim, cls_hidden_dim)
    # Loss
    self.criterion = torch.nn.CrossEntropyLoss()
    # Metrics
    self.train_correct = 0
    self.val_correct = 0
    self.test_correct = 0

  def forward(self, batch):
    out = self.model(batch)
    return out

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
    lr_scheduler = {
        'scheduler':
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2),
        'name':
            'learning_rate',
        'interval':
            'epoch',
        'frequency':
            1
    }
    return [optimizer], [lr_scheduler]
    # return optimizer

  def training_step(self, batch, batch_idx):
    out = self.forward(batch)
    loss = self.criterion(out.float(), batch.label)
    preds = torch.argmax(F.softmax(out, 1), 1)
    self.train_correct += torch.sum(preds == batch.label).item()
    self.log("loss", loss)
    self.log("train_acc_step", accuracy(preds, batch.label), prog_bar=True)
    return {'loss': loss}

  def training_epoch_end(self, outputs):
    acc = self.train_correct / self.hparams['dataset_sizes']['train']
    self.log('train_acc_epoch', acc)
    self.train_correct = 0

  def validation_step(self, batch, batch_idx):
    out = self.forward(batch)
    loss = self.criterion(out.float(), batch.label)
    preds = torch.argmax(F.softmax(out, 1), 1)
    self.val_correct += torch.sum(preds == batch.label).item()
    self.log("val_loss", loss, on_step=True, prog_bar=True)
    self.log("val_acc_step", accuracy(preds, batch.label), prog_bar=True)
    return {'val_loss': loss}

  def validation_epoch_end(self, outputs):
    acc = self.val_correct / self.hparams['dataset_sizes']['val']
    self.log('val_acc_epoch', acc)
    self.val_correct = 0

  def test_step(self, batch, batch_idx):
    out = self.forward(batch)
    loss = self.criterion(out.float(), batch.label)
    preds = torch.argmax(F.softmax(out, 1), 1)
    self.test_correct += torch.sum(preds == batch.label).item()
    self.log("test_loss", loss)
    self.log("test_acc_step", accuracy(preds, batch.label), prog_bar=True)
    return {'test_loss': loss}

  def test_epoch_end(self, outputs):
    acc = self.test_correct / self.hparams['dataset_sizes']['test']
    self.log('test_acc_epoch', acc)
    self.test_correct = 0
    print('Final Test Accuracy:', acc)


def train(args):

  print('Training arguments: ', args)

  seed_everything(args.seed)

  os.makedirs(args.log_dir, exist_ok=True)

  data = SNLIData(batch_size=args.batch_size)
  train_loader, val_loader, test_loader = data.get_iters()

  checkpoint_callback = ModelCheckpoint(monitor='val_loss')

  trainer = Trainer(
      default_root_dir=args.log_dir,
      limit_train_batches=args.
      limit_train_batches,  # for testing with less data
      fast_dev_run=False,  # for checking with 1 batch,
      callbacks=[
          LearningRateMonitor(logging_interval='step'), checkpoint_callback
      ],
      logger=TensorBoardLogger(args.log_dir, name=args.encoder_type),
      gpus=1 if torch.cuda.is_available() else 0,
      max_epochs=args.epochs,
      progress_bar_refresh_rate=args.refresh_rate)

  model = NLINet(encoder_type=args.encoder_type,
                 enc_hidden_dim=args.enc_hidden_dim,
                 cls_hidden_dim=args.cls_hidden_dim,
                 lr=args.lr,
                 dataset_sizes=data.sizes)

  # Training
  trainer.fit(model, train_loader, val_loader)
  print('Best checkpoint:', checkpoint_callback.best_model_path)
  # Testing
  #   model = NLINet.load_from_checkpoint(
  #       trainer.checkpoint_callback.best_model_path)
  test_result = trainer.test(model, test_dataloaders=test_loader, verbose=True)
  return test_result


if __name__ == '__main__':
  train(parse_args())