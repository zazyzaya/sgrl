import torch 

def make_training_split(g, tr=0.8):
    anoms = g.edge_index[:, g.y == 1]
    benign = g.edge_index[:, g.y == 0]

    num_tr = int(benign.size(1) * tr)
    idx = torch.randperm(benign.size(1))
    
    # Split benign 
    tr = benign[:, idx[:num_tr]]
    te = benign[:, idx[num_tr:]]
    
    # Add anoms back into test data 
    te = torch.cat([te, anoms], dim=1)
    y = torch.zeros(te.size(1))
    y[-anoms.size(0):] = 1

    return tr, (te,y)