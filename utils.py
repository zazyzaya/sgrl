import torch

def make_training_split(g, tr=0.8):
    anoms = g.edge_index[:, g.y == 1]
    benign = g.edge_index[:, g.y == 0]

    num_tr = int(benign.size(1) * tr)
    idx = torch.randperm(benign.size(1))
    
    # Split benign 
    tr = benign[:, idx[:num_tr]]
    te = benign[:, idx[num_tr:]]
    
    # Add anoms back into test data 
    te = torch.cat([te, anoms], dim=1)
    y = torch.zeros(te.size(1))
    y[-anoms.size(0):] = 1

    return tr, (te,y)

import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, roc_auc_score
def generate_auc_plot(clf, X_test, y_test):
    y_pred_proba = clf.predict_proba(X_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.clf()
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.savefig('auc.png')