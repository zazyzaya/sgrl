import glob 
import json 

import torch
from torch_geometric.data import Data  

from pico_globals import PICO_DIR, IP_MAP

PC = 0
USR= 1
SERVICE = 2

nmap = dict()
def get_nid(n):
    if (nid := nmap.get(n)) is None: 
        nid = len(nmap)
        nmap[n] = nid 
        ntypes.append(get_type(n))

    return nid 

ntypes = []
def get_type(n):
    if '$' in n:
        return PC
    elif 'SERVICE' in n:
        return SERVICE
    else: 
        return USR 

def parse_name(n):
    n = n.upper() 
    n = n.replace('@', '/')
    n = n.split('/')[0]

    if n in IP_MAP:
        n = IP_MAP[n]

    return n 

def parse_service(n):
    '''
    Services are weird. Normally they look something like this:
        krbtgt/g.lab
    Which is just the client authenticating w host IP 
    using the Kerb server. We don't care all that much
    about the service in that case. 
    But sometimes they look like these:
        host/HR-Win7-1.g.lab
        RPCSS/RND-WIN10-1.g.lab
        hr-win7-2$@G.LAB
        HR-WIN7-2$
        LDAP/CORP-DC.g.lab/g.lab

    And it's clients actually using a service on another machine
    These are far more informative
    '''
    n = n.upper()
    
    spl = n.split('/',1)
    if len(spl) > 1: 
        service,host = spl 
    else: 
        host = spl[0]
     
    host = host.split(".")[0] 
    host = host.split('@')[0]
    
    # Sometimes represented as user, sometimes as 
    # machine. Need to check if it's a machine
    if host + '$' in IP_MAP.values():
        host = host + '$'

    if host == 'G':
        return None 

    return host 

def load_file(f):   
    f = open(f, 'r')
    line = f.readline() 
    src,dst = [],[]

    while (line): 
        db = json.loads(line)
        
        if not ((client := db.get('client')) and \
            (service := db.get('service')) and \
            (orig_ip := db.get('id.orig_h'))):

            line = f.readline() 
            continue 

        client,orig_ip = [
            parse_name(n) for n in 
            [client,orig_ip]
        ]
        service = parse_service(service) 

        if client == 'LOCAL.ADMIN' or client == 'ADMINISTRATOR': 
            client = orig_ip

        client,orig_ip = [
            get_nid(n) for n in 
            [client, orig_ip]
        ]

        src += [orig_ip,client]
        dst += [client,orig_ip]

        if service is not None: 
            service = get_nid(service)
            src += [client,service]
            dst += [service,client]
        
        line = f.readline()

    return src,dst

def load_day(d):
    files = glob.glob(f'{PICO_DIR}/2019-07-{d}/kerberos.*')
    src,dst = [],[]

    for f in files:
        s,d = load_file(f)
        src += s 
        dst += d 

    return torch.tensor([src,dst])

def load_dataset():
    tr = load_day(19)
    te = torch.cat([load_day(i) for i in [20,21]], dim=1)

    tr = Data(
        x=torch.tensor(ntypes),
        edge_index=tr, 
        nmap=nmap 
    )

    te = Data(
        x=torch.tensor(ntypes),
        edge_index=te, 
        nmap=nmap 
    )

    return tr,te

tr_g, te_g = load_dataset()
uq_tr = tr_g.edge_index.unique(dim=1)
uq_te = te_g.edge_index.unique(dim=1) 

inv_map = {v:k for k,v in nmap.items()}
benign = set(
    [
        (
            inv_map[uq_tr[0,i].item()], 
            inv_map[uq_tr[1,i].item()]
        )
        for i in range(uq_tr.size(1))
    ]
)
maybe_red = set(
    [
        (
            inv_map[uq_te[0,i].item()], 
            inv_map[uq_te[1,i].item()]
        )
        for i in range(uq_tr.size(1))
    ]
)

for r in maybe_red:
    if r not in benign:
        print(r)