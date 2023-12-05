import glob 

from joblib import Parallel, delayed
import torch 

from csr import CSR

IN_DIR = '/mnt/raid1_ssd_4tb/datasets/LANL15/ntlm_auths'
OUT_DIR = 'databuilders/tmp'

graphs = glob.glob(f'{IN_DIR}/*.pt')

def convert(gfile):
    g = torch.load(gfile) 
    csr = CSR(g)
    
    fname = gfile.split('/')[-1].split('.')[0]
    fname = f'{OUT_DIR}/{fname}_csr.pt'
    torch.save(csr, fname)

Parallel(n_jobs=16, prefer='processes')(
    delayed(convert)(gf) for gf in graphs
)
