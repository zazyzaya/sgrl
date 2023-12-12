import glob 

import torch 
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam 

from databuilders.lanl_globals import LANL_DIR, FIRST_RED, LAST_RED, LAST_FILE
from models.simple_seal import SEAL 

tr_graphs = list(range(1,FIRST_RED)) + list(range(LAST_RED+1, LAST_FILE))
te_graphs = list(range(FIRST_RED, LAST_RED))

KHOPS = 2
LR = 0.001
EPOCHS = 1000 
BATCH_SIZE = 256

model = SEAL(KHOPS)
opt = Adam(model.parameters())
loss_fn = BCEWithLogitsLoss()
load_g = lambda x : torch.load(f'{LANL_DIR}/{x}.pt')

for e in range(EPOCHS):
    prev_graph = load_g(0).edge_index
    for i in tr_graphs:
        model.train()
        opt.zero_grad()

        query_graph = load_g(i).edge_index
        pos_query = query_graph[:, torch.randperm(query_graph.size(1))[:BATCH_SIZE]]
        full_query = torch.cat([
            pos_query, 
            torch.randint(0,query_graph.max()+1, pos_query.size())
        ], dim=1)
        labels = torch.zeros(full_query.size(0))
        labels[pos_query.size(0):] = 1

        x,ei,query = model.sample(full_query, prev_graph)
        scores = model(x,ei,query)
        loss = loss_fn(scores, labels)

        loss.backward()
        opt.step()

        print(f'[{e}-{i}] LP Loss: {loss.item()}')
        prev_graph = query_graph