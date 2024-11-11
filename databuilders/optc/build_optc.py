from collections import defaultdict
from dateutil import parser
from math import ceil
import os

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from databuilders.optc.split_optc import FIRST_TS, SNAPSHOT_SIZE

# Output of split_optc
FLOW_DIR = '/mnt/raid1_ssd_4tb/datasets/OpTC/flow_split'
USER_DIR = '/mnt/raid1_ssd_4tb/datasets/OpTC/user_split'

MAX_HOSTS = 1000 # I think it's actually less than this but it doesn't matter

# Constraining everything to be between business hours 9AM - 5PM
TRAIN = [
    ('2019-09-17T09:00:00.000-04:00','2019-09-17T17:00:00.000-04:00'),
    ('2019-09-18T09:00:00.000-04:00','2019-09-18T17:00:00.000-04:00'),
    ('2019-09-19T09:00:00.000-04:00','2019-09-19T17:00:00.000-04:00'),
    ('2019-09-20T09:00:00.000-04:00','2019-09-20T17:00:00.000-04:00')
]
TEST = [
    ('2019-09-23T09:00:00.000-04:00','2019-09-23T17:00:00.000-04:00'),
    ('2019-09-24T09:00:00.000-04:00','2019-09-24T17:00:00.000-04:00'),
    ('2019-09-25T09:00:00.000-04:00','2019-09-25T17:00:00.000-04:00')
]

def find_file_num(timecode):
    timestamp = parser.parse(timecode).timestamp()
    delta = timestamp - FIRST_TS
    assert delta >= 0, f"{timecode} is before start of records"

    fnum = int(delta) // SNAPSHOT_SIZE
    return fnum

def to_flow_file(fnum):
    return f'{FLOW_DIR}/{fnum}.csv'

def to_user_file(fnum):
    return f'{USER_DIR}/{fnum}.csv'

def find_range(start, end):
    st = find_file_num(start)
    en = find_file_num(end)
    return range(st, en+1)

def build_graph(st,en, granularity=60):
    '''
    St:             timestamp for start time
    En:             timestamp for end time
    granularity:    edges that appear within a span of `granularity` minutes
                    will be compressed, and their edge weight will be tracked
    '''
    files = find_range(st,en)

    init_ts = parser.parse(st).timestamp()
    final_ts = parser.parse(en).timestamp()
    snapshot_num = lambda x : int(x - init_ts) // (granularity*60)
    user_map = dict()

    def get_user_id(usr):
        if (usr_id := user_map.get(usr)) is None:
            usr_id = len(user_map) + MAX_HOSTS
            user_map[usr] = usr_id

        return usr_id

    def flow_to_tokens(line):
        ts,src,dst,_ = line.split(',')
        return float(ts), int(src), int(dst)

    def usr_to_tokens(line):
        ts,host,user,_ = line.split(',')
        return float(ts), int(host), get_user_id(user)

    # Read files
    edges = defaultdict(lambda : defaultdict(lambda : 0))
    for file_num in tqdm(files):
        # Parse flows
        file = to_flow_file(file_num)
        if os.path.exists(file):
            with open(file, 'r') as f:
                while (line := f.readline()):
                    ts,src,dst = flow_to_tokens(line)

                    if ts >= init_ts and ts <= final_ts:
                        edges[snapshot_num(ts)][(src,dst)] += 1

        # Parse users
        file = to_user_file(file_num)
        if os.path.exists(file):
            with open(file, 'r') as f:
                while (line := f.readline()):
                    ts,host,user = usr_to_tokens(line)

                    # Add these as undirected edges since they're so infrequent
                    if ts >= init_ts and ts <= final_ts:
                        edges[snapshot_num(ts)][(host,user)] += 1
                        edges[snapshot_num(ts)][(user,host)] += 1


    # Add edges from each snapshot
    src,dst,ts,ew = [],[],[],[]
    for i,edge in enumerate(edges.values()):
        s,d = zip(*edge.keys())
        w = list(edge.values())

        src += list(s)
        dst += list(d)
        ew += w
        ts += [i] * len(ew)

    # Use user vs computer as features
    x = torch.zeros(max(max(src), max(dst))+1, 3)
    x[:MAX_HOSTS, 0] = 1
    x[MAX_HOSTS:, 1] = 1

    # sysadmin is special, and gets its own label
    for value in ['sysadmin', 'Administrator']:
        if (id := user_map.get(value)) is not None:
            x[id] = torch.tensor([0,0,1])

    # Re-number to ignore unused nodes
    edge_index = torch.tensor([src,dst])
    uq, reindex = edge_index.unique(return_inverse=True)
    x = x[uq]

    node_names = []
    inv_usr_map = {v:k for k,v in user_map.items()}
    for idx in uq:
        if idx < MAX_HOSTS:
            node_names.append(f'Sysclient{idx.item():04}')
        else:
            node_names.append(inv_usr_map[idx.item()])

    edge_weight = torch.tensor(ew)
    ts = torch.tensor(ts)

    # Convert to torch and return
    return Data(
        x = x,
        edge_index = reindex,
        edge_weight = edge_weight,
        ts = ts,
        node_names = node_names
    )