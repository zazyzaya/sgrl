import glob 
from math import ceil 

from sklearn.metrics import roc_auc_score, average_precision_score
import torch 
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam 
from tqdm import tqdm 

from databuilders.lanl_globals import LANL_DIR, FIRST_RED, LAST_RED, LAST_FILE
import generators as g 
from models.simple_seal import SEAL 

tr_graphs = list(range(1,FIRST_RED)) #+ list(range(LAST_RED+1, LAST_FILE))
te_graphs = list(range(FIRST_RED, LAST_RED))

KHOPS = 2
LR = 0.01
EPOCHS = 1000 
BATCH_SIZE = 256
INFERENCE_BATCH_SIZE = 512
HIDDEN = 64
EMBEDDING = 32

model = SEAL(KHOPS, gnn_depth=2)
opt = Adam(model.parameters())
loss_fn = BCEWithLogitsLoss()
load_g = lambda x : torch.load(f'{LANL_DIR}/{x}.pt')

edge_sampler = g.new_link_prediction

val_old = load_g(LAST_RED+1).edge_index
val_query = load_g(LAST_RED+2).edge_index

pos,neg = edge_sampler(val_old, val_query, BATCH_SIZE)
edges = torch.cat([pos,neg], dim=1)
val = model.sample(edges, val_old)
    
def train():
    cnt = 0 
    for e in range(EPOCHS):
        prev_graph = load_g(0).edge_index
        for i in tr_graphs:
            model.train()
            opt.zero_grad()

            query_graph = load_g(i).edge_index
            pos_query, neg_query = edge_sampler(prev_graph, query_graph, BATCH_SIZE)

            full_query = torch.cat([
                pos_query, 
                neg_query
            ], dim=1)
            labels = torch.zeros(full_query.size(1),1)
            labels[:pos_query.size(0)] = 1.

            x,ei,query = model.sample(full_query, prev_graph)
            scores = model(x,ei,query)
            loss = loss_fn(scores, labels)

            loss.backward()
            opt.step()

            with torch.no_grad():
                model.eval()
                
                scores = torch.sigmoid(model(*val))
                labels = torch.zeros(scores.size())
                labels[:pos.size(1)] = 1. 

                auc = roc_auc_score(labels, scores)
                ap  = average_precision_score(labels, scores)


            print(f'[{e}-{i}] LP Loss: {loss.item()}; Val AUC: {auc:0.4f}, AP: {ap:0.4f}')
            prev_graph = query_graph

            cnt += 1
        
        '''
        if e % 10 == 0 and e:
            evaluate(model)
        '''

@torch.no_grad()
def evaluate(model):
    model.eval()

    prev_graph = load_g(te_graphs[0]-1).edge_index
    to_test = len(te_graphs)

    ys = []
    preds = []
    for i,gid in enumerate(te_graphs):
        graph = load_g(gid)
        query_edges = graph.edge_index
        
        y = torch.zeros(query_edges.size(1))
        y[graph.y_idx] = 1 
        ys.append(y)

        n_batches = ceil(query_edges.size(1) / INFERENCE_BATCH_SIZE)
        for b in tqdm(range(n_batches), desc=f'{i+1}/{to_test}'):
            st = INFERENCE_BATCH_SIZE*b 
            en = st + INFERENCE_BATCH_SIZE
            edges = query_edges[:, st:en]

            x,ei,query = model.sample(edges, prev_graph)
            labels = torch.sigmoid(model(x,ei,query))

            preds.append(labels.flatten())

        print("Stats so far")
        print(f"AUC: {roc_auc_score(torch.cat(ys), torch.cat(preds))}")
        print(f"AP : {average_precision_score(torch.cat(ys), torch.cat(preds))}")
        return

    auc = roc_auc_score(torch.cat(ys), torch.cat(preds))
    ap = average_precision_score(torch.cat(ys), torch.cat(preds))

    print("===== FINAL SCORE =====")
    print(f"AUC: {auc}")
    print(f"AP : {ap}")

    return auc,ap

if __name__ == '__main__':
    train()