import torch 
from torch.optim import Adam, SparseAdam
from torch_geometric.utils import add_remaining_self_loops, to_undirected
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm 

from models.n2v import Node2Vec, DNN
from databuilders.lanl_globals import * 
from utils import generate_auc_plot

EMB_SIZE = 128

# Generated w dataloaders.load_static_lanl
(tr_g,tr_w), (te_g,te_w,y) = torch.load('saved_graphs/lanl_static_split_c-c.pt')

ei = to_undirected(tr_g)
ei = add_remaining_self_loops(ei)[0]
n2v = Node2Vec(
    ei, 
    embedding_dim=EMB_SIZE, 
    walk_length=10, 
    context_size=5, 
    walks_per_node=20,
    sparse=True
)
n2v_opt = SparseAdam(list(n2v.parameters()), lr=0.01)
loader = n2v.loader(batch_size=128, shuffle=True)

dnn = DNN(EMB_SIZE)
dnn_opt = Adam(dnn.parameters(), lr=0.01)

weights = torch.load('saved_weights/n2v.pt')
n2v.load_state_dict(weights)

def train_n2v(e):
    # Train embedder
    total_loss = 0
    for pos_rw, neg_rw in tqdm(loader):
        n2v.train()
        n2v_opt.zero_grad()
        loss = n2v.loss(pos_rw, neg_rw)
        loss.backward()
        n2v_opt.step()
        total_loss += loss.item()

    total_loss /= len(loader)
    print(f"[{e}] Loss: {total_loss:0.4f}")

    torch.save(n2v.state_dict(), 'saved_weights/n2v.pt')

def train_log_regression(solver):
    _,neg = n2v.sample(torch.arange(tr_g.max()+1))

    neg = embed(neg[:,:2].T)
    pos = embed(tr_g)

    x = torch.cat([neg,pos])
    y = torch.zeros(x.size(0))
    y[:neg.size(0)] = 1 
    
    #weight = torch.cat([
    #    # Want neg samples to be equally weighted(?)
    #    torch.full((neg.size(0),), tr_w.mean()), 
    #    tr_w
    #])

    lr = LogisticRegression(
        max_iter=100, 
        C=(1/3)
    )
    lr.fit(x,y) #, sample_weight=weight)
    eval(lr)

bce = torch.nn.BCEWithLogitsLoss()
def train_dnn_lr():
    for e in range(100):
        dnn.train() 
        dnn_opt.zero_grad()
        _,neg = n2v.sample(torch.arange(ei.max()+1))
        neg = embed(neg[:,:2].T)
        pos = embed(tr_g)

        x = torch.cat([neg,pos])
        y = torch.zeros(x.size(0),1)
        y[:neg.size(0)] = 1 

        y_hat = dnn(x)
        loss = bce(y_hat, y)
        loss.backward() 
        dnn_opt.step()

        print(f'[{e}] Loss: {loss.item()}')

        if e % 10 == 0 and e:
            eval(dnn)
    eval(dnn)

@torch.no_grad()
def embed(g):
    n2v.eval()
    z = n2v.forward(torch.arange(g.max()+1))

    #mean = z.mean(dim=0)
    #std = z.std(dim=0) 
    #z = (z-mean) / std 

    src,dst = g 
    return z[src] * z[dst]


@torch.no_grad()
def eval(lr):
    emb = embed(te_g)
    preds = lr.predict_proba(emb)[:,1]
    classif = (preds > 0.9).astype(int)
    model_guess = lr.predict(emb)

    generate_auc_plot(lr, embed(te_g), y)

    auc = roc_auc_score(y,preds, sample_weight=te_w)
    pr = precision_score(y, classif, sample_weight=te_w)
    re = recall_score(y, classif, sample_weight=te_w)
    cls_re = recall_score(y, model_guess, sample_weight=te_w)

    # Hate working w numpy. Just flipping back to torch
    classif = torch.from_numpy(classif) 
    model_guess = torch.from_numpy(model_guess)

    thresh_fp = ((classif * (y==0).long()) * te_w).sum()
    cls_fp  = ((model_guess * (y==0).long()) * te_w ).sum()
    tn = ((y == 0).long() * te_w ).sum()
    
    thresh_fpr = thresh_fp / tn 
    cls_fpr = cls_fp / tn 

    print("Eval: ")
    print(f"\tAUC: {auc}")
    print(f"\tPr : {pr}")
    print(f"\tRe : thresh {re}, cls {cls_re}")
    print(f"\tFPR: thresh {thresh_fpr}, cls {cls_fpr}")

    with open('stats.txt', 'a') as f:
        f.write(f'{auc},{re},{thresh_fpr},{cls_re},{cls_fpr}\n')

    return auc,pr,re

with open('stats.txt', 'w+') as f:
    f.write('AUC,Thresh Re,Thresh FPR,CLS Re, CLS FPR\n')

solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
for e in range(50):
    #train_n2v(e)
    print(solvers[e])
    train_log_regression(solvers[e])
    #train_dnn_lr()