import gzip 
import torch
from torch_geometric.data import Data 
from tqdm import tqdm  

from lanl_globals import RAW_LANL, SPLIT_LANL, LAST_FILE

RED  = f'{RAW_LANL}/redteam.txt.gz'
OUT = 'tmp'

red_events = dict()
red = gzip.open(RED, 'rt')

rline = red.readline() 
while rline:
    t,usr,s,d = rline.split(',')
    d = d[:-1] # Strip newline 
    t = int(t)
    usr = usr.split('@')[0]

    if t not in red_events:
        red_events[t] = []
    
    red_events[t].append((usr,s,d))
    rline = red.readline()

src, dst = [],[]
ts = []
y_idx = []
ntypes = []

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

nmap = dict()
def get_nid(n):
    if (nid := nmap.get(n)) is None:
        nid = len(nmap)
        nmap[n] = nid 
        ntypes.append(get_ntype(n))
    return nid 

def process_line(l):
    global src,dst

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

    src += [get_nid(src_c), get_nid(src_u), get_nid(dst_u)]
    dst += [get_nid(dst_c), get_nid(src_c), get_nid(dst_c)]

def dump(f):
    global src,dst

    fname = f'{OUT}/{f}.pt'
    edge_index, weights = torch.tensor([src,dst], dtype=torch.long).unique(dim=1, return_counts=True)

    reds = []
    st = f*3600; en = (f+1)*3600 
    for k,v in red_events.items():
        if k >= st and k < en:
            for (usr,s,d) in v: 
                reds.append([get_nid(s), get_nid(d)])

    if reds: 
        reds = torch.tensor(reds).T 
        red_src = (edge_index[0] == reds[0].unsqueeze(-1))
        red_dst = (edge_index[1] == reds[1].unsqueeze(-1))
        y_idx = (red_src * red_dst).sum(dim=0).nonzero().squeeze(-1)
    else:
        y_idx = torch.tensor([])

    g = Data(
        edge_index=edge_index, 
        edge_weight=weights,
        y_idx = y_idx.long()
    )
    torch.save(g, fname)

    src,dst = [],[]


for f in tqdm(range(LAST_FILE+1)):
    cur_f = open(f'{SPLIT_LANL}/{f}.txt')
    
    line = cur_f.readline()
    while(line):
        if 'NTLM' in line.upper():
            process_line(line)
        line = cur_f.readline()

    cur_f.close()
    dump(f)