from collections import Counter
import gzip 

import torch
from torch_geometric.data import Data 
from tqdm import tqdm  

'''
Basically the same as `build_lanl_dataset`, but non-temporal
Keeps track of edge weight instead of timestamp.
All held in single file (hopefully..)
'''

# DATA = '/mnt/raid1_ssd_4tb/datasets/LANL15'
DATA = '/mnt/raid0_ssd_8tb/isaiah/LANL15'
AUTH = f'{DATA}/auth.txt.gz'
RED  = f'{DATA}/redteam.txt.gz'
#OUT = 'tmp'
OUT = f'{DATA}/ntlm_auths'

auth = gzip.open(AUTH, 'rt')
red = gzip.open(RED, 'rt')

nmap = dict()
edges = Counter()
redlist = dict()

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
    src = get_nid(l[3])
    dst = get_nid(l[4])
    edge = (src,dst)

    if l[0] == cur_red[0]:
        if  l[1] == cur_red[1] and \
            l[3] == cur_red[2] and \
            l[4] == cur_red[3]: 

            redlist[edge] = 1
            cur_red = next(red_gen)
    
    return edge 

line = auth.readline()
prog = tqdm()
dump_rate = next_dump = 60*60

def dump(edge_list): edges.update(edge_list)

edge_list = []
while(line):
    t = int(line.split(',', 1)[0])
    if t >= next_dump:
        dump(edge_list)
        edge_list = []
        next_dump += dump_rate

    if 'NTLM' in line:
        edge_list.append(process_line(line))
        prog.desc = str(t)
        prog.update()

    line = auth.readline()

dump(edge_list)

src,dst,weights,ys = [],[],[],[]

for k,v in edges.items():
    s,d = k; 
    src.append(s); dst.append(d)
    weights.append(v)

    if k in redlist: 
        ys.append(1)
    else:
        ys.append(0)

torch.save(
    Data(
        edge_index=torch.tensor([src,dst]),
        edge_attr=torch.tensor(weights),
        y=torch.tensor(ys)
    ), 
    f'{OUT}/nontemporal_ntlm.pt'
)