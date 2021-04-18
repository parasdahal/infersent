import os
import torch
from torchtext.data import Field, Iterator
from torchtext.datasets import SNLI

import spacy

spacy_en = spacy.load('en_core_web_sm')


class SNLIData():

  def __init__(self, batch_size):

    self.text = Field(
        lower=True,
        tokenize=lambda x: [tok.text for tok in spacy_en.tokenizer(x)],
        batch_first=True)
    self.label = Field(sequential=False, unk_token=None, is_target=True)

    self.train, self.dev, self.test = SNLI.splits(self.text, self.label)
    self.sizes = {
        'train': len(self.train),
        'val': len(self.dev),
        'test': len(self.test)
    }
    self.text.build_vocab(self.train, self.dev)
    self.label.build_vocab(self.train)

    vector_cache_loc = '.vector_cache/snli_vectors.pt'
    if os.path.isfile(vector_cache_loc):
      self.text.vocab.vectors = torch.load(vector_cache_loc)
    else:
      self.text.vocab.load_vectors('glove.840B.300d')
      torch.save(self.text.vocab.vectors, vector_cache_loc)

    # Batching
    self.train_iter, self.dev_iter, self.test_iter = Iterator.splits(
        (self.train, self.dev, self.test),
        batch_size=batch_size,
        device='cuda:0' if torch.cuda.is_available() else 'cpu')

    self.vocab_size = len(self.text.vocab)
    self.out_dim = len(self.label.vocab)
    self.labels = self.label.vocab.stoi

  def get_iters(self):
    return self.train_iter, self.dev_iter, self.test_iter

  def get_vocab(self):
    return self.text.vocab.stoi