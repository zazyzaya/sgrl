import numpy as np 
import torch 
from torch_geometric.utils import to_dense_adj, dense_to_sparse, remove_self_loops
from databuilders.lanl_globals import LANL_NUM_NODES

def negative_sampling(pos_sample, batch_size, num_nodes, oversample=1.25): 
    ei_hash = lambda x : x[0, :] + x[1, :] * num_nodes
    
    ei1d = ei_hash(pos_sample).numpy()
    neg = np.array([[],[]])

    while neg.shape[1] < batch_size:
        maybe_neg = np.random.randint(0,num_nodes+1, (2, int(batch_size*oversample)))
        maybe_neg = maybe_neg[:, maybe_neg[0] != maybe_neg[1]] # Remove self-loops
        neg_hash = ei_hash(maybe_neg)

        neg = np.concatenate(
            [neg, maybe_neg[:, ~np.in1d(neg_hash, ei1d)]],
            axis=1 
        )

    neg = neg[:, :batch_size]
    return torch.tensor(neg).long()


def link_prediction(old,new, batch_size, num_nodes=LANL_NUM_NODES):
    pos = remove_self_loops(new)[0]
    neg = negative_sampling(old, batch_size, num_nodes)
    pos = pos[:, torch.randperm(pos.size(1))[:batch_size]]

    return pos,neg
    

def new_link_prediction(old,new, batch_size, num_nodes=LANL_NUM_NODES):
    local_num_nodes = 1+max(old.max(), new.max()) 
    dense_old = to_dense_adj(old, max_num_nodes=local_num_nodes)[0].bool()
    dense_new = to_dense_adj(new, max_num_nodes=local_num_nodes)[0].bool()

    new_links = ((~dense_old).logical_and(dense_new)).long()

    # Remove self loops 
    new_links[torch.arange(local_num_nodes), torch.arange(local_num_nodes)] = 0 
    new_links = dense_to_sparse(new_links)[0]
    new_links = new_links[:, torch.randperm(new_links.size(1))[:batch_size]]

    neg = negative_sampling(old, batch_size, num_nodes)

    return new_links, neg