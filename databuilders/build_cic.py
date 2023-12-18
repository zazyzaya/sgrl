import glob 

from tqdm import tqdm 
import torch 
from torch_geometric.data import Data

from cic_globals import *

nmap = dict() 
def get_nid(ip):
    if (nid := nmap.get(ip)) is None:
        nid = len(nmap) 
        nmap[ip] = nid 

    return nid 

def file_to_graph(fname):
    f = open(fname, 'r', errors='replace')
    f.readline() # Discard header

    line = f.readline()

    name = FMAP[fname.split('/')[-1]]
    prog = tqdm(desc=name)

    src, dst, ts, ys = [],[],[],[]
    last_s = None; last_d = None
    i = 0 
    t0 = None 
    while(line):
        tokens = line.split(',')
        s = tokens[1]; d = tokens[3]; 
        t = tokens[6]; y = tokens[-1][:-1]

        if s == '' or d == '':
            line = f.readline()
            continue 

        s = get_nid(s); d = get_nid(d) 
        if s == last_s and d == last_d:
            line = f.readline() 
            continue 

        last_s = s; last_d = d

        t = t.split(' ')[1] # Always same day so dont care about date
        t_tokens = t.split(':') # [Hour, Minute, Second]
        t = sum([int(t_tokens[i])*(60**(2-i)) for i in range(2)])

        if t0 is None: 
            t0 = t 

        t -= t0  

        src.append(s)
        dst.append(d)
        ts.append(t) 

        if y != 'BENIGN': 
            ys.append(i) 

        i += 1 
        line = f.readline() 
        prog.update()

    return Data(
        edge_index = torch.tensor([src,dst]),
        ts = torch.tensor(ts),
        y_idx = torch.tensor(ys) 
    )

if __name__ == '__main__':
    files = glob.glob(f'{CIC_HOME}/*.csv')
    for f in files:
        data = file_to_graph(f)
        out_f = FMAP[f.split('/')[-1]]

        torch.save(data, f'tmp/{out_f}.pt')

    torch.save(nmap, 'tmp/nmap.pt')