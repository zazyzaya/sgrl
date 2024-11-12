from databuilders.optc.build_optc import build_graph, TRAIN, TEST
import torch


def pos_sample(hosts):
    pass


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