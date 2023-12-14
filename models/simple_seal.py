import torch 
from torch import nn
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models import GCN

class SEAL(nn.Module):
    '''
    Completely hangs on dense graphs (e.g. LANL)
    The paper talks about needing to use 1-hop for 
    a dataset w 10k edges, or they had OOM errors.. 
    '''
    def __init__(self, khops, gnn_depth=None, embed_size=32, hidden=64):
        super().__init__()
        self.khops = khops 
        self.dist_mp = MessagePassing(aggr='max')

        self.enc_dim = self.__drnl_idx(
            *torch.tensor([khops*2]).repeat(2).split(1)
        )+1
        self.hidden_dim = hidden 
        self.embed_size = embed_size

        gnn_depth = khops if gnn_depth is None else gnn_depth
        self.model = GCN(
            self.enc_dim, 
            self.hidden_dim, 
            gnn_depth, 
            out_channels=embed_size,
            dropout=0.1
        )
        #self.out_net = nn.Linear(embed_size, 1)

    def __drnl_idx(self, dx,dy):
        d = dx+dy
        idx = 1+torch.min(dx,dy) + d/2 * (d/2 + d%2 - 1)
        idx = idx.nan_to_num(0,0,0)
        
        # Need to do some cleanup
        idx[torch.isinf(d)] = 0 
        idx[d <= 1] = 1

        return idx.long().flatten()

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
            n, ei, x_y, _ = k_hop_subgraph(
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
        dist = torch.zeros(offset, 2)
        dist[targets[:, 0], 0] = self.khops+1
        dist[targets[:, 1], 1] = self.khops+1

        for _ in range(self.khops):
            d = self.dist_mp.propagate(sgs, x=dist-1)
            dist = torch.max(dist,d)

        # Convert s.t. starting nodes are 0, one hop is 1 etc 
        # Unreached nodes are khops+1
        dist = (dist-self.khops + 1)
        dist[dist == 0] = torch.inf 

        labels = self.dnrl(*dist.split(1,dim=1))
        return labels, sgs, targets 
    
    def forward(self, x, ei, targets):
        '''
        Takes the output of self.sample and passes it through a GNN
        '''
        z = self.model(x,ei)
        src = z[targets[0]]
        dst = z[targets[1]]

        return (src*dst).sum(dim=1, keepdim=True)

if __name__ == '__main__':
    '''
        0   1   2
         \ / \ /
          3   4
         / \ / \ 
        5   6   7
    '''
    ei = torch.tensor([
        [0,3], [3,0], [3,1], [1,3], [1,4], [4,1], 
        [4,2], [2,4], [5,3], [3,5], [3,6], [6,3], 
        [6,4], [4,6], [4,7], [7,4]
    ]).T 
    seal = SEAL(1)

    seal.sample(torch.tensor([[3,4], [0,3]]), ei)