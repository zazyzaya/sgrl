import glob 

from tqdm import tqdm 
import torch
from databuilders.lanl_globals import * 

def load():
    tr_edges = torch.tensor([[],[]])
    te_edges = torch.tensor([[],[]])
    y = torch.tensor([])
    
    for idx in tqdm(range(LAST_FILE+1)):
        g = torch.load(f"{LANL_DIR}/{idx}.pt")

        # Train
        if g.y_idx.size(0) == 0:
            tr_edges = torch.cat([
                tr_edges, g.edge_index
            ], dim=1)
            tr_edges = tr_edges.unique(dim=1)
        
        # Test
        else:
            te_edges = torch.cat([
                te_edges, g.edge_index
            ], dim=1)
            g_y = torch.zeros(g.edge_index.size(1))
            g_y[g.y_idx] = 1
            y = torch.cat([y, g_y])

    return tr_edges, (te_edges, y)