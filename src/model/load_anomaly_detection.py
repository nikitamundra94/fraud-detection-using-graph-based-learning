import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random


def load_anomaly_detection_dataset():
    
    data_mat = sio.loadmat('BlogCatalog.mat')
    adj = data_mat['Network']
    feat = data_mat['Attributes']
    #feat = feat[:,0:10]
    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_norm = adj_norm.toarray()
    adj = adj + sp.eye(adj.shape[0])
    adj = adj.toarray()
    feat = feat.toarray()
    truth = data_mat['Label']
    truth = truth.flatten()
    return adj_norm, adj, feat, truth 



def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()