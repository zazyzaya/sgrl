import numpy as np 
import torch
from torch_geometric.nn.models import GCN 
from torch_geometric.utils import k_hop_subgraph

from databuilders.lanl_globals import LANL_NUM_NODES
from models.simple_seal import SEAL

'''
Implementing GraIL from:
https://proceedings.mlr.press/v119/teru20a/teru20a.pdf

Very similar to SEAL but uses the intersect of node-pair subgraphs 
rather than the union 
'''

class GraIL(SEAL):
    def __init__(self, khops, gnn_depth=None, embed_size=32, hidden=64, num_nodes=LANL_NUM_NODES):
        super().__init__(khops, gnn_depth, embed_size, hidden)
        
        self.num_nodes = num_nodes
        self.noninductive = False # This can be a param eventually

        if self.noninductive: 
            gnn_depth = khops if gnn_depth is None else gnn_depth

            self.model = GCN(
                self.enc_dim + num_nodes, 
                self.hidden_dim, 
                gnn_depth, 
                out_channels=embed_size,
                dropout=0.1
            )

    def sample(self, query_edges, edge_index):
        num_sample_nodes = max(query_edges.max()+1, edge_index.max()+1)
        sgs = []
        ids = []
        offset = 0
        targets = []

        # Other than this for-loop pruning non-shared nodes, identical to SEAL 
        for i in range(query_edges.size(1)): 
            n_src, src_sg, src, mask_src = k_hop_subgraph(
                query_edges[0,i:i+1], 
                self.khops, 
                edge_index,
                num_nodes=num_sample_nodes
            )
            src = n_src[src]

            n_dst, dst_sg, dst, mask_dst = k_hop_subgraph(
                query_edges[1,i:i+1],
                self.khops, 
                edge_index, 
                num_nodes=num_sample_nodes
            )
            dst = n_dst[dst]

            n_src = torch.cat([n_src, dst]) # Make sure dst intersects w src nodes
            n_dst = torch.cat([n_dst, src])
            intersect = torch.from_numpy(np.intersect1d(n_src, n_dst)).view(-1,1)
            edges = edge_index[:, mask_src + mask_dst]

            # Search for locations in edges that contain a shared node 
            mask = (
                # Create |intersect| copies of edges looking for each uq node
                (intersect - edges.view(-1)).transpose(-1,-2) == 0
            # Then sum together to find final output
            ).sum(dim=-1).view(edges.size()).sum(dim=0).bool() 
            
            uq,ei = edges[:, mask].unique(return_inverse=True)

            # Need to find src/dst's new id's 
            new_nodes = uq.size(0)
            x = (uq == src).nonzero()
            y = (uq == dst).nonzero()

            if x.size(0) == 0:
                x = torch.tensor([[new_nodes]])
                uq = torch.cat([uq, x.squeeze(-1)])
                new_nodes += 1 
            if y.size(0) == 0: 
                y = torch.tensor([[new_nodes]])
                uq = torch.cat([uq, y.squeeze(-1)])
                new_nodes += 1 

            x_y = torch.cat([x,y])

            sgs.append(ei+offset)
            targets.append(x_y+offset)
            ids.append(uq)
            offset += new_nodes

        sgs = torch.cat(sgs, dim=1)
        targets = torch.cat(targets, dim=1)
        ids = torch.cat(ids)

        # Matrix of distance to x or y 
        dist = torch.zeros(offset, 2)
        dist[targets[0], 0] = self.khops+1
        dist[targets[1], 1] = self.khops+1

        for _ in range(self.khops):
            d = self.dist_mp.propagate(sgs, x=dist-1)
            dist = torch.max(dist,d)

        # Convert s.t. starting nodes are 0, one hop is 1 etc 
        # Unreached nodes are khops+1
        dist = dist - (self.khops + 1)
        dist *= -1 
        dist = dist.long()
        #dist[dist == (self.khops+1)] = torch.inf 

        # labels = self.dnrl(*dist.split(1,dim=1))
        labels = self.one_hot_node_distance(*dist.split(1,dim=1))

        if self.noninductive: 
            labels = torch.cat([
                labels, 
                torch.eye(self.num_nodes)[ids]
            ], dim=1)

        return labels, sgs, targets 
    
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
    grail = GraIL(2)
    grail.sample(torch.tensor([[0], [2]]), ei)