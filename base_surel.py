import torch 
from torch_cluster import random_walk
from tqdm import tqdm 

from math import ceil

from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn 
from torch.optim import Adam
from tqdm import tqdm 

from utils import make_training_split
from models.surel import get_walks, SUREL
from databuilders.lanl_globals import LANL_DIR

'''
PyTorch implimentation of SUREL: 
https://arxiv.org/pdf/2202.13538.pdf
'''

WL = 3
NW = 200
embed_dim = 64
epochs = 10_000
patience = 5 # Same as paper 
lr = 1e-3
bs = 1_500
inference_bs = 1_500

DEVICE = 3
g = torch.load(f'{LANL_DIR}/nontemporal_ntlm.pt')

# TODO validation set also 
tr_ei, (te_ei, te_y) = make_training_split(g)
H,T,walks = get_walks(tr_ei, wl=WL, nwalks=NW)
T = T.to(DEVICE)

model = SUREL(WL, embedding_size=embed_dim, device=DEVICE)
opt = Adam(model.parameters(), lr=lr)
nbatches = ceil(tr_ei.size(1) / bs)
loss_fn = nn.BCEWithLogitsLoss()

for e in range(epochs):
    # Shuffle edges 
    model.train()
    tr_ei = tr_ei[:, torch.randperm(tr_ei.size(1))]

    pos = tr_ei[:, :bs]
    neg = torch.randint(0, tr_ei.max()+1, (2,pos.size(1)))

    pos_score = model(H,T,walks,*pos)
    neg_score = model(H,T,walks,*neg)
    scores = torch.cat([pos_score, neg_score])

    y = torch.ones(pos_score.size(0)*2,1, device=DEVICE)
    y[pos_score.size(0):] = 0

    loss = loss_fn(scores, y)
    loss.backward()
    opt.step() 

    print(f'[{e}] {loss.item()}')

    if e % 99 == 0 and e:
        model.eval()
        with torch.no_grad():
            te_score = []
            for i in tqdm(range(ceil(te_ei.size(1) / inference_bs)), desc='Testing...'):
                st = inference_bs*i; en = (i+1)*inference_bs
                edges = te_ei[:, st:en]
                if edges.size(1):
                    te_score.append(model(H,T,walks, *edges))
            
            te_score = 1-torch.sigmoid(torch.cat(te_score)).cpu()
            print(f"AUC: {roc_auc_score(te_y, te_score)}")
            print(f"AP:  {average_precision_score(te_y, te_score)}")
            
            with open('tracker.txt', 'a+') as f: 
                f.write(f"[{e}]\n")
                f.write(f"AUC: {roc_auc_score(te_y, te_score)}\n")
                f.write(f"AP:  {average_precision_score(te_y, te_score)}\n\n")
                    
    torch.save(model.state_dict(), 'weights.pt')