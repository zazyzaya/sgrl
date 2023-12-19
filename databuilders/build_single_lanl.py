from collections import Counter
import gzip 

import torch
from torch_geometric.data import Data 
from tqdm import tqdm  

from lanl_globals import RAW_LANL

'''
Basically the same as `build_lanl_dataset`, but non-temporal
Keeps track of edge weight instead of timestamp.
All held in single file (hopefully..)
'''

AUTH = f'{RAW_LANL}/auth.txt.gz'
RED  = f'{RAW_LANL}/redteam.txt.gz'
OUT = 'tmp'
#OUT = f'{LANL_DIR}/ntlm_auths'

auth = gzip.open(AUTH, 'rt')
red = gzip.open(RED, 'rt')

ntypes = []
nmap = dict()
edges = Counter()
redlist = dict()

USR = 0
COM = 1
ANO = 3 
ETC = 2

get_ntype = lambda x : \
    USR if x.startswith('U') \
    else COM if x.startswith('C') \
    else ANO if x.startswith('ANON') \
    else ETC # I don't think there's anything else, but just in case

def parse_str(n):
    nt = get_ntype(n) 
    
    # Users and computers
    # Users formatted as U123@DOM1
    # Computers as C123$@DOM1 (if src node, else just C123)
    if nt < 2: 
        n  = n.split('@')[0]
        if nt == 1:
            n = n.replace('$', '')
    
    return n, nt

def get_nid(n):
    if not (nid := nmap.get(n)):
        nid = len(nmap)
        nmap[n] = nid 
        ntypes.append(get_ntype(n))

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

def process_line_old(l):
    global cur_red

    l = l.split(',')
    src_u, sut = parse_str(l[1])
    src_c, sct = parse_str(l[3])
    dst_u, dut = parse_str(l[2])
    dst_c, dct = parse_str(l[4])

    # Anonymous logins are essentially computers talking to 
    # computers, so only care about src machine 
    # It's always(?) ANON@C123 -> C123 otherwise
    if sut == ANO: 
        src = get_nid(src_c)
    else: 
        src = get_nid(src_u)

    edge = (src, get_nid(dst_c))

    if l[0] == cur_red[0]:
        if  l[1] == cur_red[1] and \
            l[3] == cur_red[2] and \
            l[4] == cur_red[3]: 

            redlist[edge] = 1
            cur_red = next(red_gen)
    
    return edge

def process_line(l):
    global cur_red

    l = l.split(',')
    src_u, sut = parse_str(l[1])
    src_c, sct = parse_str(l[3])
    dst_u, dut = parse_str(l[2])
    dst_c, dct = parse_str(l[4])

    # Want to make the following edges from each line:
    # SRC       DST
    #  u         u
    #  |         |
    #  c ------- c 

    edges = [
        (get_nid(src_u), get_nid(src_c)),
        (get_nid(dst_u), get_nid(dst_c)),
        (get_nid(src_c), get_nid(dst_c))
    ]

    if l[0] == cur_red[0]:
        if  (l[1] == cur_red[1] or l[2] == cur_red[1]) and \
            l[3] == cur_red[2] and \
            l[4] == cur_red[3]: 

            redlist[edges[-1]] = 1
            cur_red = next(red_gen)
    
    return edges

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
        edge_list += process_line(line)
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
        x=torch.tensor(ntypes),
        edge_index=torch.tensor([src,dst]),
        edge_attr=torch.tensor(weights),
        y=torch.tensor(ys)
    ), 
    f'{OUT}/nontemporal_ntlm.pt'
)