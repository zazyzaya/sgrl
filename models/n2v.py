from typing import Optional
import torch
from torch import Tensor, nn 
from torch_geometric.nn import Node2Vec

class LogRegression(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.l1 = nn.Linear(embedding_size, 1)

    def forward(self, z):
        return self.l1(z)