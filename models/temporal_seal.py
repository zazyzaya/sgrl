import torch 
from torch_geometric.utils import k_hop_subgraph

from models.simple_seal import SEAL
from utils import k_hop_csr 

class TemporalSEAL(SEAL):
    def sample(self, query_edges, query_times, csr): 
        sgs = []
        offset = 0
        targets = []

        # There must be a way to parallelize this... 
        for i in range(query_edges.size(1)): 
            n, ei, _, x_y = k_hop_csr(
                query_edges[:,i], 
                query_times,
                self.khops, 
                csr
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