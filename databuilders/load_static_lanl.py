import glob 
from math import ceil

from tqdm import tqdm 
import torch
from lanl_globals import * 


def load_day(i):
    st = i*24; en = min((i+1)*24, LAST_FILE+1)
    
    edges = torch.tensor([[],[]], dtype=torch.long)
    y = torch.tensor([])
    for i in range(st,en):
        g = torch.load(f"{LANL_DIR}/{i}.pt")
        g.y_idx = g.y_idx.long() 

        edges = torch.cat([
            edges, g.edge_index
        ], dim=1)
        y_ = torch.zeros(g.edge_index.size(1))
        y_[g.y_idx] = 1

    if y.sum(): 
        return edges, y, True 
    else: 
        return edges.unique(dim=1), y, False 
    
def load():
    tr_edges = torch.tensor([[],[]], dtype=torch.long)
    te_edges = torch.tensor([[],[]], dtype=torch.long)
    y = torch.tensor([])
    
    for idx in tqdm(range(ceil(LAST_FILE/24))):
        edges, y_, is_test = load_day(idx)
        if is_test:
            te_edges = torch.cat([
                te_edges, edges
            ], dim=1)
            y = torch.cat([y, y_])

        else: 
            tr_edges = torch.cat([
                tr_edges, edges
            ], dim=1)

    tr_edges = tr_edges.unique(dim=1)
    return tr_edges.long(), (te_edges.long(), y.long())

if __name__ == '__main__':
    out = load()
    torch.save(out, 'tmp/lanl_static_split.pt')