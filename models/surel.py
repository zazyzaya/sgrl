import torch 
from torch import nn
from torch_cluster import random_walk
from tqdm import tqdm 

def get_walks(ei, wl=4, nwalks=200):
    '''
    Paper recommends wl in (2, 5), nwalks in (50, 400)
    '''
    start = torch.arange(ei.max()+1).repeat_interleave(nwalks)
    walks = random_walk(
        *ei, start, wl-1
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
    T = T / nwalks  # Normalize between 0 and 1 

    # Ensure T[-1] == 0 so we can use -1 for nodes without relative embeddings
    T = torch.cat([T, torch.zeros(1, T.size(1))])

    H = []
    i = 0
    for walk_set in tqdm(h_keys, desc='Mapping X to H'):
        h = dict()
        for s in walk_set:
            h[s.item()] = h_vals[i].item()
            i += 1 
        H.append(h)

    return H, T, walks_per_node

class SUREL(nn.Module):
    def __init__(self, wl, embedding_size=64, device='cpu'):
        super().__init__()
        self.embedder = nn.Sequential(
            nn.Linear(wl*2, embedding_size*2, device=device), 
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(embedding_size*2, embedding_size, device=device),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.LSTM(embedding_size, embedding_size, 2, device=device)
        )

        self.lp = nn.Sequential(
            nn.Linear(embedding_size, embedding_size, device=device), 
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(embedding_size, 1, device=device)
        )

    def embed(self, H,T,walks, src,dst):
        batch = torch.cat([src,dst])
        batch_size = batch.size(0)

        # Get repr of nodes from src perspective
        u = walks[batch]
        [
            u[i].apply_(lambda x : H[s].get(x,-1))
            for i,s in enumerate(src.repeat(2))
        ]                   # Remap each node to T idxs 
        u = T[u]    # batch x num_walks x wl x d

        # Get repr of nodes from dst perspective
        v = walks[batch]
        [
            v[i].apply_(lambda x : H[d].get(x,-1))
            for i,d in enumerate(dst.repeat(2))
        ] 
        v = T[v]  # batch x num_walks x wl x T.dim(-1)

        # Stick node embeddings together       
        x = torch.cat([u, v], dim=-1)

        # Combine batch and walks into single dim
        x = x.reshape(
            batch_size * x.size(1), # batch * num_walks
            x.size(2),              # Walk length
            x.size(3)               # Embedding size (2*wl)
        )

        # Transpose for LSTM optimization (sequence order in 0th dim)
        x = x.transpose(0,1)    # wl x batch*n walks*2 x d
        z,_ = self.embedder(x)
        z = z[-1]               # Only care about last output 

        # Separate back into batches
        z = z.reshape(
            batch_size, 
            z.size(0) // batch_size, 
            z.size(-1)
        )

        # Mean pool each LSTM walk embedding per-batch
        z = z.mean(dim=1)  # batch_size x dim 

        # Top half were src nodes, bottom half were dst nodes
        # (for link prediction tasks)
        z_src, z_dst = z.split(z.size(0) // 2) 
        return z_src, z_dst

    def forward(self, H,T,walks, src,dst):
        z_src, z_dst = self.embed(H,T,walks,src,dst)

        # Original paper just combines the walks together
        # in the mean-pooling part (they're considered additional
        # walk samples all together). So we'll just avg them together
        # too. 
        z = (z_src + z_dst)/2
        return self.lp(z)