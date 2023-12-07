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

'''
PyTorch implimentation of SUREL: 
https://arxiv.org/pdf/2202.13538.pdf
'''

WL = 4
NW = 50
embed_dim = 64
epochs = 10_000
patience = 5 # Same as paper 
lr = 1e-3
bs = 256

INPUT = '/mnt/raid1_ssd_4tb/datasets/LANL15/ntlm_auths'
g = torch.load(f'{INPUT}/nontemporal_ntlm.pt')

# TODO validation set also 
tr_ei, (te_ei, te_y) = make_training_split(g)
H,T,walks = get_walks(tr_ei, wl=WL, nwalks=NW)

model = SUREL(WL, embedding_size=embed_dim)
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

        pos_score = model(H,T,walks,*pos)
        neg_score = model(H,T,walks,*neg)
        scores = torch.cat([pos_score, neg_score])

        y = torch.zeros(pos_score.size(0)*2,1)
        y[pos_score.size(0):] = 1

        loss = loss_fn(scores, y)
        loss.backward()
        opt.step() 

        print(f'[{e}-{b}] {loss.item()}')

        if b % 100 == 0 and b:
            model.eval()
            with torch.no_grad():
                te_score = model(H,T,walks, *te_ei)
                print(f"AUC: {roc_auc_score(te_y, te_score)}")
                print(f"AP:  {average_precision_score(te_y, te_score)}")
                
                with open('tracker.txt', 'a+') as f: 
                    f.write(f"[{e}-{b}]\n")
                    f.write(f"AUC: {roc_auc_score(te_y, te_score)}\n")
                    f.write(f"AP:  {average_precision_score(te_y, te_score)}\n\n")
                    

    model.eval()
    with torch.no_grad():
        te_score = model(H,T,walks, *te_ei)
        print(f"AUC: {roc_auc_score(te_y, te_score)}")
        print(f"AP:  {average_precision_score(te_y, te_score)}")