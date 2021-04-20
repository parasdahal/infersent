import sys
import numpy as np
import argparse

# import SentEval
sys.path.insert(0, './SentEval')

import spacy
import torch
from pytorch_lightning import Trainer

from train import NLINet
from data import SNLIData

spacy_en = spacy.load('en_core_web_sm')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def senteval(checkpoint_path, params_senteval):
  import senteval
  model = NLINet.load_from_checkpoint(checkpoint_path).to(device)
  model.eval()
  data = SNLIData(batch_size=128)

  def prepare(params, samples):
    params.vocab = data.get_vocab()
    params.max_len = np.max([len(x) for x in samples])
    params.wvec_dim = 300
    return

  def batcher(params, batch):
    samples = []
    for sent in batch:
      sent_idxs = []
      for token in sent:
        sent_idxs.append(params.vocab[token])
      # padding
      for _ in range(len(sent_idxs) + 1, params.max_len + 1):
        sent_idxs.append(params.vocab["<pad>"])
      samples.append(sent_idxs)

    embed = torch.LongTensor(samples).to(device)
    return model.model.encode(embed).detach().cpu().numpy()

  se = senteval.engine.SE(params_senteval, batcher, prepare)

  transfer_tasks = [
      'MR',
      'CR',
      'SUBJ',
      'MPQA',
      'SST2',
      'TREC',
      'MRPC',
      'SICKEntailment',
  ]

  results = se.eval(transfer_tasks)
  print(results)
  return results


def process_senteval_result(result_dict):
  devaccs = []
  ndevs = []
  for task, res in result_dict.items():
    try:
      devaccs.append(res['devacc'])
      ndevs.append(res['ndev'])
    except:
      pass
  print('Macro accuracy:', np.mean(devaccs))
  macros = [(ndevs[i] / sum(ndevs)) * devaccs[i] for i in range(len(devaccs))]
  print('Micro accuracy:', sum(macros))


def snli(checkpoint_path):
  data = SNLIData(batch_size=128)
  _, _, test_loader = data.get_iters()
  model = NLINet.load_from_checkpoint(checkpoint_path).to(device)
  model.eval()
  trainer = Trainer(weights_summary=None)
  test_result = trainer.test(model,
                             test_dataloaders=test_loader.to(device),
                             verbose=True)
  print(test_result)
  return test_result


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--checkpoint_path',
                      type=str,
                      help='Path to model checkpoint')
  args = parser.parse_args()
  params_senteval = {
      'task_path': './SentEval/data/',
      'usepytorch': False,
      'kfold': 5,
      'classifier': {
          'nhid': 0,
          'optim': 'rmsprop',
          'batch_size': 128,
          'tenacity': 3,
          'epoch_size': 2
      }
  }
  print("######### Evaluating SNLI #########")
  snli(args.checkpoint_path)

  print("######### Evaluating SENTEVAL #########")
  result_dict = senteval(args.checkpoint_path, params_senteval)
  process_senteval_result(result_dict)
