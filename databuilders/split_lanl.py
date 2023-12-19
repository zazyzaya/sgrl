import gzip 
from tqdm import tqdm 
from lanl_globals import * 

AUTH = f'{LANL_DIR}/auth.txt.gz'
file = gzip.open(AUTH,'rt')

HOUR = 3600
OUT = 'tmp' 

line = file.readline() 
cur_ts = 0 
out_f = open(f'{OUT}/0.txt', 'w+')

prog = tqdm(total=LAST_FILE)
while(line):
    ts,_ = line.split(',', 1)
    ts = int(ts) 

    if ts > cur_ts + HOUR:
        out_f.close()
        out_f = open(f'{OUT}/{ts // HOUR}.txt', 'w+')
        cur_ts += HOUR 
        prog.update()

    out_f.write(line)
    line = file.readline()

out_f.close()