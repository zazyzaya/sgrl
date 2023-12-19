import torch 
from torch.optim import Adam 
from torch_geometric.utils import add_remaining_self_loops, to_undirected
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

from models.n2v import Node2Vec, LogRegression
from databuilders.lanl_globals import * 
from utils import generate_auc_plot

# Generated w dataloaders.load_static_lanl
tr_g, (te_g,y) = torch.load('saved_graphs/lanl_static_split.pt')

ei = to_undirected(tr_g)
ei = add_remaining_self_loops(ei)[0]
n2v = Node2Vec(ei, 128, 10, 5, 20)
n2v_opt = Adam(n2v.parameters(), lr=0.01)

#weights = torch.load('saved_weights/n2v.pt')
#n2v.load_state_dict(weights)

def train_n2v(epochs=10):
    # Train embedder
    for e in range(epochs):
        pos,neg = n2v.sample(torch.arange(te_g.max()+1))

        n2v.train()
        n2v_opt.zero_grad()
        n2v_loss = n2v.loss(pos,neg)
        n2v_loss.backward()
        n2v_opt.step()

        print(f"[{e}] Loss: {n2v_loss.item():0.4f}")

    torch.save(n2v.state_dict(), 'saved_weights/n2v.pt')

def train_log_regression():
    _,neg = n2v.sample(torch.arange(ei.max()+1).repeat(5))
    neg = embed(neg[:, :2].T)
    pos = embed(tr_g)

    x = torch.cat([neg,pos])
    y = torch.zeros(x.size(0))
    y[:neg.size(0)] = 1 

    lr = LogisticRegression()
    lr.fit(x,y)
    eval(lr)


@torch.no_grad()
def embed(g):
    n2v.eval()
    z = n2v.forward(torch.arange(g.max()+1))
    src,dst = g 
    return z[src] * z[dst]


@torch.no_grad()
def eval(lr):
    preds = lr.predict_proba(embed(te_g))[:,1]
    classif = (preds > 0.9).astype(int)

    generate_auc_plot(lr, embed(te_g), y)

    auc = roc_auc_score(y,preds)
    pr = precision_score(y, classif)
    re = recall_score(y, classif)

    fpr = ((classif == 1) * (y.numpy() == 0)).sum() / classif.sum()

    print("Eval: ")
    print(f"\tAUC: {auc}")
    print(f"\tPr : {pr}")
    print(f"\tRe : {re}")
    print(f"\tFPR: {fpr}")

    return auc,pr,re

for _ in range(10):
    train_n2v()
    train_log_regression()