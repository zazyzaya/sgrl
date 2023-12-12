import torch 
from torch import nn
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models import GCN

class SEAL(nn.Module):
    '''
    Completely hangs on dense graphs (e.g. LANL)
    '''
    def __init__(self, khops, embed_size=32, hidden=64):
        super().__init__()
        self.khops = khops 
        self.dist_mp = MessagePassing(aggr='sum')

        self.enc_dim = self.__drnl_idx(
            *torch.tensor([khops*2]).repeat(2).split(1)
        )+1
        self.hidden_dim = hidden 
        self.embed_size = embed_size

        self.model = GCN(
            self.enc_dim, 
            self.hidden_dim, 
            khops, 
            out_channels=embed_size,
            dropout=0.1
        )
        self.out_net = nn.Linear(embed_size, 1)

    def __drnl_idx(self, dx,dy):
        d = dx+dy
        idx = 1+torch.min(dx,dy) + d/2 * (d/2 + d%2 - 1)
        idx = idx.nan_to_num(0)
        return idx.long()

    def dnrl(self, dx,dy):
        out = torch.zeros(dx.size(0), self.enc_dim)
        idx = self.__drnl_idx(dx,dy)

        out[torch.arange(dx.size(0)), idx] = 1.
        return out 
    
    def sample(self, query_edges, edge_index): 
        sgs = []
        offset = 0
        targets = []

        # There must be a way to parallelize this... 
        for i in range(query_edges.size(1)): 
            n, ei, _, x_y = k_hop_subgraph(
                query_edges[:,i], 
                self.khops, 
                edge_index,
                relabel_nodes=True,
                num_nodes=max(query_edges.max()+1, edge_index.max()+1)
            )
            sgs.append(ei+offset)
            targets.append((x_y+offset).unsqueeze(-1))
            offset += n.size(0)

        sgs = torch.cat(sgs, dim=1)
        targets = torch.cat(targets, dim=1)

        # Matrix of distance to x or y 
        dist = torch.full((offset, 2), torch.inf)
        dist[targets[0], 0] = 0
        dist[targets[1], 1] = 0

        for _ in range(self.khops):
            d = 1 + self.dist_mp.propagate(sgs, x=dist)
            dist = torch.min(dist, d)

        labels = self.dnrl(*dist.split(1,dim=1))

        return labels, sgs, targets 
    
    def forward(self, x, ei, targets):
        '''
        Takes the output of self.sample and passes it through a GNN
        '''
        z = self.model(x,ei)
        src = z[targets[0]]
        dst = z[targets[1]]

        return self.out_net(src*dst)