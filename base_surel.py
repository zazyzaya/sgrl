import torch 
from torch_cluster import random_walk
from tqdm import tqdm 


'''
PyTorch implimentation of SUREL: 
https://arxiv.org/pdf/2202.13538.pdf
'''


INPUT = '/mnt/raid1_ssd_4tb/datasets/LANL15/ntlm_auths'

def get_walks(g, wl=10, nwalks=10):
    start = torch.arange(g.edge_index.max()+1).repeat_interleave(nwalks)
    walks = random_walk(
        *g.edge_index, start, wl-1
    ) 

    # Split into per-node batches
    walks_per_node = walks.reshape(
        walks.size(0) // nwalks, 
        nwalks, 
        wl
    )

    X = []
    h_keys = []
    # Prob a way to parallelize this
    for walk in tqdm(walks_per_node, desc='Building X'):
        s,reidx = walk.unique(return_inverse=True)
        x = torch.zeros(s.size(0), wl)
        ones = torch.ones(reidx.size(0), wl)
        x.scatter_add_(0,reidx,ones)

        X.append(x)
        h_keys.append(s)
    
    print("Pruning X")
    T,h_vals = torch.cat(X, dim=0).unique(dim=0, return_inverse=True)
    
    H = []
    i = 0
    for walk_set in tqdm(h_keys, desc='Mapping X to H'):
        h = dict()
        for s in walk_set:
            h[s.item()] = h_vals[i].item()
            i += 1 
        H.append(h)

    return H, T 

g = torch.load(f'{INPUT}/nontemporal_ntlm.pt')
X,T = get_walks(g)