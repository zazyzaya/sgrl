import socket 
env = socket.gethostname()

RAW_LANL = '/mnt/raid1_ssd_4tb/datasets/LANL15'
SPLIT_LANL = f'{RAW_LANL}/split_auth_raw'

if env == 'SEAS14012':
    LANL_DIR = 'lanl_data/'
else:
    LANL_DIR = '/mnt/raid1_ssd_4tb/datasets/LANL15/ntlm_auths'
    
LAST_FILE = 1389
FIRST_RED = 41
LAST_RED = 711
LANL_NUM_NODES = 15609