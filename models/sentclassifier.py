import torch
import torch.nn as nn


class SentClassifier(nn.Module):

  def __init__(self, input_dim, hidden_dim, out_dim, batch_size, device):

    super(SentClassifier, self).__init__()
    self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
		self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.lin3 = nn.Linear(self.dim, self.out_dim)
    self.relu = nn.ReLU()
    
    self.network = nn.Sequential(self.lin1, self.relu, self.lin2, self.relu, self.lin3)

  def forward(self, premise, hypothesis):
    combined = torch.cat((premise, hypothesis, torch.abs(premise - hypothesis), premise * hypothesis),1)
    out = self.network(combined)
    return out