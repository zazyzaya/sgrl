import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.nn.models import GIN
from torch_scatter import segment_csr

class MultiPool(nn.Module):
    '''
    Does std, max, min, and mean pooling
    '''
    N_POOLS = 3

    def forward(self, embs, batches, lens):
        '''
        Parameters:
            embs:   N x d input matrix
            batches: List of idxs in each batch
            lens:   CSR-style length of each batch. E.g. [0,2,5] means that
                    the batches are batches[0:2], batches[2:5]
        '''
        src = embs[batches]

        return torch.cat([
            #segment_csr(src, lens, reduce='std'),
            segment_csr(src, lens, reduce='mean'),
            segment_csr(src, lens, reduce='min'),
            segment_csr(src, lens, reduce='max')
        ], dim=1)


class GLASS(nn.Module):
    def __init__(self, in_dim, hidden, layers=3, lr=0.001):
        super().__init__()

        self.gnn = GIN(in_dim+1, hidden, layers)
        self.pool = MultiPool()
        self.readout = nn.Sequential(
            nn.Linear(hidden * self.pool.N_POOLS, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        self.loss = nn.BCEWithLogitsLoss()
        self.opt = Adam(self.parameters(), lr=lr, weight_decay=0.001)

        self.args = [in_dim, hidden]
        self.kwargs = dict(layers=layers, lr=lr)

    def save(self, out_f):
        torch.save([self.args, self.kwargs, self.state_dict()], out_f)

    def predict(self, x,ei,ew, batches, return_logits=False):
        lens = [0]
        for b in batches:
            lens.append(b.size(0) + lens[-1])

        lens = torch.tensor(lens)
        batches = torch.cat(batches)

        # GLASS contribution: add a feature for nodes in the batches.
        batch_feat = torch.zeros(x.size(0), 1)
        batch_feat[batches] = 1
        x = torch.cat([x, batch_feat], dim=1)

        z = self.gnn(x, ei, edge_weight=ew)
        z = self.pool(z, batches, lens)
        logits = self.readout(z)

        if return_logits:
            return logits
        else:
            return 1 / (1+torch.exp(-logits))

    def forward(self, x,ei,ew, batches, labels):
        self.opt.zero_grad()
        preds = self.predict(x,ei,ew, batches, return_logits=True)
        loss = self.loss.forward(preds, labels)
        loss.backward()
        self.opt.step()

        return loss