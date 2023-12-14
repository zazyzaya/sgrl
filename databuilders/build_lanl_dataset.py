import gzip 
import torch
from torch_geometric.data import Data 
from tqdm import tqdm  

DATA = '/mnt/raid1_ssd_4tb/datasets/LANL15'
AUTH = f'{DATA}/auth.txt.gz'
RED  = f'{DATA}/redteam.txt.gz'
OUT = 'tmp'

auth = gzip.open(AUTH, 'rt')
red = gzip.open(RED, 'rt')
etypes = {'U':0, 'C':1, 'A':2}

src, dst = [],[]
edge_attr, ts = [],[]
y_idx = []

nmap = dict()
def get_nid(n):
    if not (nid := nmap.get(n)):
        nid = len(nmap)
        nmap[n] = nid 
    return nid 

def red_event():
    line = red.readline()
    while(line):
        t,usr,s,d = line.split(',')
        d = d[:-1] # Strip newline 
        yield t,usr,s,d 

        line = red.readline()

red_gen = red_event()
cur_red = next(red_gen)

def process_line(l):
    global cur_red

    l = l.split(',')
    ts.append(int(l[0]))
    edge_attr.append(etypes.get(l[1][0], 2))
    src.append(get_nid(l[3]))
    dst.append(get_nid(l[4]))

    if l[0] == cur_red[0]:
        if  l[1] == cur_red[1] and \
            l[3] == cur_red[2] and \
            l[4] == cur_red[3]: 

            y_idx.append(len(src))
            cur_red = next(red_gen)

def dump():
    global src,dst,ts,edge_attr,y_idx

    fname = f'{OUT}/{ts[0]//(60*60)}.pt'
    g = Data(
        edge_index = torch.tensor([src,dst], dtype=torch.long),
        ts=torch.tensor(ts),
        edge_attr=torch.tensor(edge_attr),
        y_idx=torch.tensor(y_idx)
    )
    torch.save(g, fname)

    src,dst,ts,edge_attr,y_idx = [],[],[],[],[]

line = auth.readline()
prog = tqdm()
dump_rate = next_dump = 60*60

while(line):
    t = int(line.split(',', 1)[0])
    if t >= next_dump:
        dump()
        next_dump += dump_rate

    if 'NTLM' in line:
        process_line(line)
        prog.desc = str(ts[-1])
        prog.update()

    line = auth.readline()

dump()