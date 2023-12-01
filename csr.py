from collections.abc import Iterable

import torch 
import torch_geometric
from torch_geometric.utils import sort_edge_index
from tqdm import tqdm 

class CSR():
    def __init__(self, g):
        self.to_csr(g)
        self.verbose = True 

    def to_csr(self, g):
        ei,(ts,ew) = sort_edge_index(
            g.edge_index, 
            edge_attr=[g.ts, g.edge_attr]
        ) # type: ignore

        ptr = [0]
        cur = 0 
        for i in tqdm(range(ei.size(1)), disable=not self.verbose):
            if ei[0][i] != cur: 
                ptr.append(i)
                cur = ei[0][i].item() 

        self.ptr = torch.tensor(ptr, dtype=torch.long)
        self.idx = ei[1]
        self.ts = ts 
        self.ew = ew 

    def __len__(self): 
        return self.idx.size(0)
    
    def __get_one(self, idx):
        st = self.ptr[idx]; en = self.ptr[idx+1]
        return self.idx[st:en], self.ts[st:en], self.ew[st:en]

    def __getitem__(self, key):
        if isinstance(key, Iterable):
            eis, ts, ews = zip(
                *[self.__get_one(i) for i in key]
            )
            return \
                torch.cat(eis), \
                torch.cat(ts), \
                torch.cat(ews)
        
        return self.__get_one(key)