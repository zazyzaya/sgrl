from typing import Optional
import torch
from torch import Tensor, nn 
from torch_geometric.nn import Node2Vec

class DNN(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 1)
        )

    def forward(self, z):
        return self.l1(z)

    @torch.no_grad()
    def predict_proba(self, z):
        p = torch.sigmoid(self.l1(z))
        return torch.cat([
            1-p, p
        ], dim=1).numpy() 
    
    @torch.no_grad()
    def predict(self, z):
        p = torch.sigmoid(self.l1(z)).squeeze(-1)
        return (p > 0.5).long().numpy()
