from random import choice

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.utils import add_remaining_self_loops

from databuilders.featurizer import add_features
from models.glass import GLASS
from samplers import pos_sample, neg_sample
from databuilders.optc.get_labels import OPTC_LABELS

HOST_FEAT = 0
BATCH_SIZE = 64
EPOCHS = 10_000

def edge_weight_norm(x, ei, ew):
    '''
    Two kinds of edge weights, HOST to HOST
    and HOST to USER. Normalize each seperately
    '''
    h2h = torch.logical_and(
        x[ei[0]][:, HOST_FEAT] == 1,
        x[ei[1]][:, HOST_FEAT] == 1
    )

    ew = ew.float()
    h2h_e = ew[h2h]
    h2h_e = (h2h_e - h2h_e.mean()) / (h2h_e.std() + 1e8)

    h2u_e = ew[~h2h]
    h2u_e = (h2u_e - h2u_e.mean()) / (h2u_e.std() + 1e8)

    ew[h2h] = h2h_e
    ew[~h2h] = h2u_e

    return ew

@torch.no_grad()
def test(model: GLASS):
    model.eval()
    log = []
    for i in range(len(OPTC_LABELS)):
        g = torch.load(f'graphs/optc/test_{i}.pt')
        ew = edge_weight_norm(g.x, g.edge_index, g.edge_weight)

        preds = []
        labels = []
        red_hosts = []
        for t in range(g.ts.max()+1):
            # As more hosts become infected, add them to the list
            # of labeled bad hosts
            new_red = OPTC_LABELS[i].get(t, [])
            for red in new_red:
                nid = g.node_names.index(f'Sysclient{red:04}')
                red_hosts.append(nid)

            ei_t = g.edge_index[:, g.ts == t]
            ew_t = ew[g.ts == t]
            ei_t,ew_t = add_remaining_self_loops(ei_t, edge_attr=ew_t, fill_value=1)


            nodes = ei_t.unique()
            x = add_features(g.x[nodes], ei_t)
            hosts = nodes[x[:, HOST_FEAT] == 1]

            # Build out list of labeled hosts
            y = []
            for h in hosts:
                if h in red_hosts:
                    y.append(1)
                else:
                    y.append(0)
            labels += y

            batches = hosts.chunk(hosts.size(0) // 16)

            # Do this one at a time for optimal scoring
            for b in batches:
                targets = pos_sample(b, ei_t)
                pred = model.predict(x, ei_t, ew_t, targets)
                preds += pred.squeeze().tolist()

        auc = roc_auc_score(labels, preds)
        print(f'AUC: {auc}')
        ap = average_precision_score(labels, preds)
        print(f'AP:  {ap}')
        log.append((auc,ap))

    return log

def step(e, g, model: GLASS):
    model.train()
    ew = edge_weight_norm(g.x, g.edge_index, g.edge_weight)

    # For now split this across timesteps. May find a better
    # way to do this later. But for now, discretize
    times = g.ts.unique()
    times = times[torch.randperm(times.size(0))]
    for t in times:
        mask = g.ts == t
        ei_t = g.edge_index[:, mask]
        ew_t = ew[mask]
        ei_t,ew_t = add_remaining_self_loops(ei_t, edge_attr=ew_t, fill_value=1)

        nodes = ei_t.unique()
        x = add_features(g.x[nodes], ei_t)
        hosts = nodes[x[:, HOST_FEAT] == 1]
        batches = torch.randperm(hosts.size(0)).split(BATCH_SIZE)

        for batch in batches:
            targets = hosts[batch]
            pos = pos_sample(targets, ei_t)
            neg = neg_sample(targets, ei_t)
            labels = torch.ones(len(pos) + len(neg), 1)
            labels[:len(pos)] = 0

            loss = model.forward(x, ei_t, ew_t, pos+neg, labels)

        print(f'[{e}-{t}] {loss.item()}')

def train(model):
    gs = [f'graphs/optc/train_{i}.pt' for i in range(4)]
    log = []

    for e in range(1,EPOCHS+1):
        gf = choice(gs)
        g = torch.load(gf)
        step(e,g,model)

        if e % 10 == 0:
            metrics = test(model)
            log.append(metrics)
            torch.save(log, 'log.pt')

        model.save('model_weights/glass.pt')

if __name__ == '__main__':
    model = GLASS(4, 32, lr=0.0001)
    train(model)
    #args,kwargs,sd = torch.load('model_weights/glass.pt')
    #model = GLASS(*args, **kwargs)
    #model.load_state_dict(sd)

    test(model)