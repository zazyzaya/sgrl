import glob 
from math import ceil

from tqdm import tqdm 
import torch
from lanl_globals import * 


#LANL_DIR += '/../ntlm_auths_c-c'

def load_day(i):
    st = i*24; en = min((i+1)*24, LAST_FILE+1)
    
    edges = torch.tensor([[],[]], dtype=torch.long)
    weights = torch.tensor([])
    y = torch.tensor([])
    for i in range(st,en):
        g = torch.load(f"{LANL_DIR}/{i}.pt")

        edges = torch.cat([
            edges, g.edge_index
        ], dim=1)
        y_ = torch.zeros(g.edge_index.size(1))
        y_[g.y_idx] = 1

        weights = torch.cat([weights, g.edge_weight])
        y = torch.cat([y, y_])

    if y.sum(): 
        return edges, y, True, weights
    else: 
        return edges.unique(dim=1), y, False, weights
    
def load():
    tr_edges = torch.tensor([[],[]], dtype=torch.long)
    tr_weight = torch.tensor([])
    te_edges = torch.tensor([[],[]], dtype=torch.long)
    te_weight = torch.tensor([])
    y = torch.tensor([])
    
    for idx in tqdm(range(ceil(LAST_FILE/24))):
        edges, y_, is_test, weight = load_day(idx)
        if is_test:
            te_edges = torch.cat([
                te_edges, edges
            ], dim=1)
            y = torch.cat([y, y_])
            te_weight = torch.cat([te_weight,weight])

        else: 
            tr_edges = torch.cat([
                tr_edges, edges
            ], dim=1)
            tr_weight = torch.cat([tr_weight, weight])

    tr_edges,tr_inv= tr_edges.unique(dim=1, return_inverse=True)
    te_edges,te_inv = te_edges.unique(dim=1, return_inverse=True)

    # Any repeated edges that were omitted, 
    # add their edge weights together
    new_tr_weight = torch.zeros(tr_edges.size(1))
    new_tr_weight.scatter_add_(-1, tr_inv, tr_weight)

    new_te_weight = torch.zeros(te_edges.size(1))
    new_te_weight.scatter_add_(-1, te_inv, te_weight)

    # Need to relabel after getting unique edges
    new_y = torch.zeros(te_edges.size(1))
    red_idx = te_inv[y.nonzero().squeeze(-1)]
    new_y[red_idx] = 1

    return (tr_edges.long(), new_tr_weight), (te_edges.long(), new_te_weight, new_y.long())

if __name__ == '__main__':
    out = load()
    torch.save(out, '../saved_graphs/lanl_static_split.pt')