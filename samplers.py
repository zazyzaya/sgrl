import torch
from torch_cluster.rw import random_walk
from torch_geometric.utils import add_remaining_self_loops

from databuilders.optc.build_optc import build_graph, TRAIN, TEST


def pos_sample(hosts, edge_index, n_walks=200, walk_len=1):
    ei = add_remaining_self_loops(edge_index)[0]
    sgs = random_walk(
        *ei, hosts.repeat_interleave(n_walks),
        walk_length=walk_len
    )

    sgs = sgs.reshape(hosts.size(0), (walk_len+1) * n_walks)
    sgs = [sg.unique() for sg in sgs]
    return sgs

def neg_sample(hosts, edge_index, n_walks=10, walk_len=1, percent_swapped=0.75):
    sgs = pos_sample(hosts, edge_index, n_walks, walk_len)
    reorder = torch.randperm(len(sgs))

    permutated = []
    for i,sg in enumerate(sgs):
        # Take 25% of original sg
        orig = sg[torch.rand(sg.size(0)) > percent_swapped]

        # And 75% of swapped sg
        to_swap = sgs[reorder[i]]
        to_swap = to_swap[to_swap != hosts[reorder[i]]] # Remove original hub node
        swapped = to_swap[torch.rand(to_swap.size(0)) <= percent_swapped]

        # Then stick them together for the new subgraph grouping
        new_rw = torch.cat([orig, swapped, hosts[i].unsqueeze(-1)]).unique()
        permutated.append(new_rw)

    return permutated


def build_optc():
    for i,t in enumerate(TRAIN):
        print(f"Train graph {i}")
        g = build_graph(*t)
        print(g.edge_index.size())
        torch.save(g, f'graphs/optc/train_{i}.pt')

    for i,t in enumerate(TEST):
        print(f"Test graph {i}")
        g = build_graph(*t)
        print(g.edge_index.size())
        torch.save(g, f'graphs/optc/test_{i}.pt')