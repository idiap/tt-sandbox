# -----------------------------------------------------------------------------
# License-Identifier: GPL-3.0-only
# This file is part of the TT-sandbox project.
# Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# Contributor: Teng Xue <teng.xue@idiap.ch>
# -----------------------------------------------------------------------------

import numpy as np

### The complete TT decomposion of a tensor
def tt_svd_full(tensor):
    ndim = len(tensor.shape)
    prev_core = None
    tt_cores = []
    pre_tensor = tensor.copy()
    for i in range(ndim):
        #unfolding tensor
        mode_size = pre_tensor.shape[i]
        prev_rank = prev_core.shape[2] if prev_core is not None else 1
        X = np.reshape(tensor, (prev_rank*mode_size, -1)) #(r_{i-1}*n_i) x  (n_{i+1}*n_{i+2}*n_{i+3]...)
        U, S, V = np.linalg.svd(X, full_matrices=False) #matrix SVD decomposition 
        core = U.reshape((prev_rank, mode_size, -1)) #r_{i-1} x n_i x r_i
        print(f"The ranks of complete {i}-th core is {core.shape}")
        tensor = np.dot(np.diag(S), V) #r_i x (n_{i+1}*n_{i+2}*n_{i+3] ...})
        if tensor.size==1: #The last core should be scaled with the scalar computed from S*V
            core = core * tensor
        tt_cores.append(core)
        prev_core = core
        
    return tt_cores

### The truncated TT decomposion based on predefined ranks
def tt_svd_rank(tensor, ranks):
    ndim = len(tensor.shape)
    prev_core = None
    tt_cores = []
    pre_tensor = tensor.copy()
    for i in range(ndim):
        #unfolding tensor
        mode_size = pre_tensor.shape[i] #tensor dimension
        prev_rank = prev_core.shape[2] if prev_core is not None else 1
        X = np.reshape(tensor, (prev_rank*mode_size, -1)) #(r_{i-1}*n_i) x  (n_{i+1}*n_{i+2}*n_{i+3]...)
        U, S, V = np.linalg.svd(X, full_matrices=False) #matrix SVD decomposition 
        # print(f"The ranks of non-truncated {i}-th core is {prev_rank, mode_size, S.shape[0]}")
        if i <ndim-1: #for the last core, the last dimension has to be 1, therefore no need to truncate ranks.
            r = min(ranks[i], S.shape[0]) #truncate the rank of cores
        else:
            r = S.shape[0]
        core = U[:, :r].reshape((prev_rank, mode_size, -1)) #r_{i-1} x n_i x r_i
        print(f"The ranks of truncated {i}-th core is {core.shape}")
        tensor = np.dot(np.diag(S[:r]), V[:r, :]) #r_i x (n_{i+1}*n_{i+2}*n_{i+3] ...})
        if tensor.size==1: #The last core should be scaled with the scalar computed from S*V
            core = core * tensor
        tt_cores.append(core)
        prev_core = core
    return tt_cores

### The truncated TT decomposition based on eighenvalues threshold
def tt_svd_thres(tensor, threshold):
    ndim = len(tensor.shape)
    prev_core = None
    tt_cores = []
    pre_tensor = tensor.copy()
    for i in range(ndim):
        #unfolding tensor
        mode_size = pre_tensor.shape[i] #tensor dimension
        prev_rank = prev_core.shape[2] if prev_core is not None else 1
        X = np.reshape(tensor, (prev_rank*mode_size, -1)) #(r_{i-1}*n_i) x  (n_{i+1}*n_{i+2}*n_{i+3]...)
        U, S, V = np.linalg.svd(X, full_matrices=False) #matrix SVD decomposition 
        r = np.sum(S>threshold) #find the index bigger than threshold
        assert r>0, "The threshold is too low" 
        core = U[:, :r].reshape((prev_rank, mode_size, -1)) #r_{i-1} x n_i x r_i
        print(f"The ranks of truncated {i}-th core is {core.shape}")
        tensor = np.dot(np.diag(S[:r]), V[:r, :]) #r_i x (n_{i+1}*n_{i+2}*n_{i+3] ...})
        if tensor.size==1: #The last core should be scaled with the scalar computed from S*V
            core = core * tensor
        tt_cores.append(core)
        prev_core = core
    return tt_cores

# reconstruct tensor given tt cores
def tt_svd_recon(tt_cores):
    tensor = tt_cores[0]
    for i in range(1, len(tt_cores)):
        tensor = np.tensordot(tensor, tt_cores[i], axes=1)
    return tensor

### 2D matrix SVD decomposition
def svd_full(X):

    """
    Truncate the ranks below the threshold
    """
    U, S, V = np.linalg.svd(X, full_matrices=False) #matrix SVD decomposition 
    
    return U, S, V

### Truncate eighenvalues given predefined threshold
def svd_thres(X, threshold):

    """
    Truncate the ranks below the threshold
    """
    U, S, V = np.linalg.svd(X, full_matrices=False) #matrix SVD decomposition 
    r = np.sum(S>threshold) #find the index bigger than threshold
    assert r>0, "The threshold is too low" 

    return U[:, :r], S[:r], V[:r, :]

### Truncate S matrix given predefined ranks
def svd_rank(X, rank):

    """
    Truncate the matrix with predefined rank
    """
    U, S, V = np.linalg.svd(X, full_matrices=False) #matrix SVD decomposition 
    r = rank
    assert r>0, "The threshold is too low" 
    return U[:, :r], S[:r], V[:r, :]

### reconstruct 2D matrix given U, S, V matrix
def svd_recon(U, S, V):
    reconstructed_tensor = np.dot(U, np.dot(np.diag(S), V))
    return reconstructed_tensor
