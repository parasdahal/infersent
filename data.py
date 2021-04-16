import os
import torch
from torchtext.data import Field, Iterator, TabularDataset
from torchtext.datasets import SNLI

class SNLIData():

  def __init__(self, batch_size):
    
    self.text = Field(lower=True, tokenize='spacy', batch_first=True)
    self.label = Field(sequential=False, unk_token=None, is_target=True)
    
    self.train, self.dev, self.test = datasets.SNLI.splits(self.text, self.label)
		
		self.text.build_vocab(self.train, self.dev)
		self.label.build_vocab(self.train)
    
    if os.path.exists('./logs/snli_glove.pt'):
      self.text.vocab.vectors = torch.load('./logs/snli_glove.pt')
    else:
      # Save vectors
      self.text.vocab.load_vectors('glove.840B.300d')
      torch.save(self.text.vocab.vectors, './logs/snli_glove.pt')
    
    self.train_iter, self.dev_iter, self.test_iter = Iterator.splits((self.train, self.dev, self.test), batch_size=batch_size, device=device)
    
    self.vocab_size = len(self.text.vocab)
    self.out_dim = len(self.label.vocab)
    self.labels = self.label.vocab.stoi
    
    return self.train_iter, self.dev_iter, self.test_iter
    
