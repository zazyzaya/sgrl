import torch 
from torch.optim import Adam 
from torch_geometric.utils import add_remaining_self_loops, to_undirected
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

from models.n2v import Node2Vec, LogRegression
from databuilders.lanl_globals import * 
from databuilders.load_static_lanl import load 

g = torch.load(f'{LANL_DIR}/nontemporal_ntlm.pt')

tr_g, (te_g,y) = torch.load('saved_graphs/lanl_static_split.pt')

ei = to_undirected(tr_g)
ei = add_remaining_self_loops(ei)[0]
n2v = Node2Vec(ei, 128, 10, 5, 20)
n2v_opt = Adam(n2v.parameters(), lr=0.01)


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


def train_log_regression():
    _,neg = n2v.sample(torch.arange(ei.max()+1))
    neg = embed(neg[:, :2].T)
    pos = embed(tr_g)

    x = torch.cat([neg,pos])
    y = torch.zeros(x.size(0))
    y[pos.size(0):] = 1 

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
    classif = lr.predict(embed(te_g))
    preds = lr.predict_proba(embed(te_g))[:,1]

    auc = roc_auc_score(y,preds)
    pr = precision_score(y, classif)
    re = recall_score(y, classif)

    print("Eval: ")
    print(f"\tAUC: {auc}")
    print(f"\tPr : {pr}")
    print(f"\tRe : {re}")

    return auc,pr,re

for _ in range(10):
    train_n2v()
    train_log_regression()