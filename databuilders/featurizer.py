import torch
from torch_ppr import page_rank

def add_features(x,ei):
    pagerank = page_rank(edge_index=ei)
    return torch.cat([x, pagerank.unsqueeze(-1)], dim=1)