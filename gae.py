import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score
import torch 
from torch import nn 
from torch.optim import Adam 
from torch_geometric.nn import GCN, MessagePassing

tr_g, (te_g, y) = torch.load('saved_graphs/lanl_static_split_c-u.pt')
x = torch.eye(tr_g.max()+1)

class AggGAE(nn.Module):
    def __init__(self, in_dim, hidden, layers, out_dim, **kwargs):
        super().__init__()

        self.gcn = GCN(in_dim, hidden, layers, out_dim, **kwargs)
        #self.mp = MessagePassing(aggr='mean')
        #self.out = nn.Linear(out_dim, in_dim)
        #self.sm = nn.Softmax(dim=1)

    def forward(self, x, ei):
        return self.gcn(x,ei)
    
    def embed(self, z, ei, query):
        return 1-torch.sigmoid(
            (z[query[0]] * z[query[1]]).sum(dim=1)
        )

model = AggGAE(x.size(1), 128, 2, 128, dropout=0.1)
opt = Adam(model.parameters(), lr=0.01)

criterion = torch.nn.BCELoss()
def train():
    for e in range(10_000):
        idx = torch.randperm(
            tr_g.size(1)
        )
        val_size = int(tr_g.size(1) * 0.1)

        val = tr_g[:, idx[:val_size]]
        tr = tr_g[:, idx[val_size:]]
        labels = torch.zeros(val.size(1)*2)
        labels[val.size(1):] = 1

        model.train() 
        opt.zero_grad()
        z = model.forward(x, tr)
        
        query = torch.cat([
            val, torch.randint(0, x.size(0), val.size())
        ], dim=1)
        preds = model.embed(z, tr, query)

        loss = criterion(preds, labels) 
        loss.backward()
        opt.step()

        if e and e % 10 == 0: 
            eval(model) 

        print(f"[{e}] Loss: {loss.item():0.4f}")
        torch.save(model.state_dict(), 'saved_weights/gae.pt')

def plot_auc(preds, y):
    fpr, tpr, _ = roc_curve(y,  preds)
    auc = roc_auc_score(y, preds)
    
    plt.clf()
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.savefig('auc.png')

    return auc 

@torch.no_grad()
def eval(model):
    model.eval() 
    z = model(x, tr_g) 

    preds = model.embed(z, tr_g, te_g)
    classif = (preds > 0.9).long() 

    auc = plot_auc(preds, y)
    pr = precision_score(y, classif)
    re = recall_score(y, classif)

    fpr = (classif == 1).logical_and(y == 0).sum() / (y==0).sum() 

    print("Eval: ")
    print(f"\tAUC: {auc}")
    print(f"\tPr : {pr}")
    print(f"\tRe : thresh {re}")
    print(f"\tFPR: thresh {fpr}")

    with open('gae_stats.txt', 'a+') as f:
        f.write(f'{auc},{re},{fpr}\n')

    return auc,pr,re

with open('gae_stats.txt', 'w+') as f:
    f.write('AUC,TPR,FPR\n')
train() 