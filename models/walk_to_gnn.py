import torch 
from torch_cluster import random_walk 

from .surel import get_walks

def rw_to_eis(walks): 
    '''
    Expects N x NW x WL tensor as input
    Outputs N x 2 x NW*WL tensor 
    '''
    src = walks[:,:,:-1]
    dst = walks[:,:,1:]

    ei = torch.stack([src,dst], dim=1)
    ei = ei.reshape(walks.size(0), 2, walks.size(1)*(walks.size(2)-1))
    return ei 

def build_disjoint_subgraphs(batch, eis):
    '''
    Take b-dimensional tensor batch
    Use it to index N x 2 x NW*(WL-1) dim edge indexes
    '''
    eis = eis[batch]            # B x 2 x d 
    eis = eis.transpose(0,1)    # 2 x B x d (all src nodes on first, dst on second) 
    
    # Probably very time consuming.. 
    # Should find more effective way to do this
    uq_ei = [
        ei.unique(return_inverse=True) for ei in eis
    ]
    uq,eis_ = zip(*uq_ei)
    eis = []
    offset = 0
    for i in range(len(eis)):
        eis.append(eis_.pop(0) + offset)
        offset += uq[i].size(0)

    eis = torch.cat(eis, dim=1)
    return torch.cat(uq), eis 



if __name__ == '__main__':
    walk = torch.tensor([
        [
            [0,1,2,3],
            [0,3,2,1]
        ],
        [
            [1,0,5,9],
            [1,2,3,0]
        ],
        [
            [4,6,7,4],
            [4,7,10,6]
        ]
    ])

    eis = rw_to_eis(walk)
    print(eis)