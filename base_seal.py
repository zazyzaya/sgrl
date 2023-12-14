from math import ceil

from sklearn.metrics import roc_auc_score, average_precision_score
import torch 
from torch import nn 
from torch.optim import Adam
from tqdm import tqdm 
from torch_geometric.utils import degree

from utils import make_training_split
from models.simple_seal import SEAL
from databuilders.lanl_globals import LANL_DIR

'''
PyTorch implimentation of SEAL: 
https://arxiv.org/pdf/1802.09691.pdf
'''

epochs = 10_000
patience = 5 # Same as paper 
lr = 1e-3
bs = 256
khops = 1

g = torch.load(f'{LANL_DIR}/nontemporal_ntlm.pt')

print(f'Highest degree: {degree(g.edge_index[0]).max()/g.edge_index.size(1)}')

# TODO validation set also 
tr_ei, (te_ei, te_y) = make_training_split(g)

model = SEAL(khops=khops)
opt = Adam(model.parameters(), lr=lr)
nbatches = ceil(tr_ei.size(1) / bs)
loss_fn = nn.BCEWithLogitsLoss()

for e in range(epochs):
    # Shuffle edges 
    tr_ei = tr_ei[:, torch.randperm(tr_ei.size(1))]
    for b in range(nbatches): 
        model.train()
        opt.zero_grad()
        st = b*bs; en = (b+1)*bs
        
        pos = tr_ei[:, st:en]
        neg = torch.randint(0, tr_ei.max()+1, (2,pos.size(1)))
        query_edges = torch.cat([pos,neg], dim=1)

        labels,subgraphs,targets = model.sample(query_edges, tr_ei)
        scores = model.forward(labels, subgraphs, targets)

        y = torch.zeros(pos.size(0)*2,1)
        y[pos.size(0):] = 1

        loss = loss_fn(scores, y)
        loss.backward()
        opt.step() 

        print(f'[{e}-{b}] {loss.item()}')

        '''
        if b % 100 == 0 and b:
            model.eval()
            with torch.no_grad():
                te_score = model(*test_params)
                print(f"AUC: {roc_auc_score(te_y, te_score)}")
                print(f"AP:  {average_precision_score(te_y, te_score)}")
                
                with open('tracker.txt', 'a+') as f: 
                    f.write(f"[{e}-{b}]\n")
                    f.write(f"AUC: {roc_auc_score(te_y, te_score)}\n")
                    f.write(f"AP:  {average_precision_score(te_y, te_score)}\n\n")
                    

    model.eval()
    with torch.no_grad():
        te_score = model(*test_params)
        print(f"AUC: {roc_auc_score(te_y, te_score)}")
        print(f"AP:  {average_precision_score(te_y, te_score)}")
    '''