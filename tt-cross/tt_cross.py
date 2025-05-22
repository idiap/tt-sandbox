# -----------------------------------------------------------------------------
# License-Identifier: GPL-3.0-only
# This file is part of the TT-sandbox project.
# Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# Contributor: Teng Xue <teng.xue@idiap.ch>
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr, lu

def maxvol(a, ind=None, niters=100, eps=5e-2):
    """
    Find the row indices of the maximal volume submatrix in a tall matrix.
    
    Parameters:
    a     -- The input matrix (n-by-r where n > r).
    ind   -- Initial row indices (optional). If not provided, LU decomposition will be used.
    niters -- Maximum number of iterations (default: 100).
    eps    -- Convergence threshold (default: 5e-2).
    
    Returns:
    ind   -- Row indices that form the maximal volume submatrix.
    """
    n, r = a.shape  # Get dimensions of matrix a

    # If n <= r, return all rows
    if n <= r:
        return np.arange(n)

    # If 'ind' is not provided, use LU decomposition to initialize 'ind'
    if ind is None:
        # Perform LU decomposition
        P, L, U = lu(a)  # LU decomposition without permute_l=True
        # Get the permutation matrix from P (the row swaps)
        p = np.argmax(P, axis=1)  # P is a permutation matrix
        ind = np.argsort(p)[:r]  # Initialize 'ind' with the first r rows of the LU pivot
    else:
        # Ensure 'ind' is a numpy array (in case an integer or list is passed)
        ind = np.asarray(ind)

    # Construct the submatrix formed by the initial row indices and ensure it's 2D
    sbm = np.atleast_2d(a[ind, :])

    # Ensure sbm is a square matrix and perform right division to compute matrix b
    if sbm.shape[1] == r:  # Ensure sbm has the correct dimensions (sbm should be r x r)
        try:
            b = np.linalg.solve(sbm.T, a.T).T  # Solve the system of linear equations
        except np.linalg.LinAlgError:
            # If singular, use least squares instead
            b = np.linalg.lstsq(sbm.T, a.T, rcond=None)[0].T
    else:
        raise ValueError(f"sbm must have {r} columns, got shape {sbm.shape}")

    # Start iterations
    iter = 0
    while iter <= niters:
        # Find the maximum absolute value in the matrix b
        mx0 = np.max(np.abs(b))
        big_ind = np.argmax(np.abs(b))

        # Get the row and column indices of the maximum element in b
        i0, j0 = np.unravel_index(big_ind, (n, r))

        # If the maximum element in b is close to 1, assume convergence and return the result
        if mx0 <= 1 + eps:
            ind = np.sort(ind)  # Sort the indices before returning
            return ind

        # Update the indices and adjust matrix b
        k = ind[j0]  # Ensure ind is always an array, and we index it correctly
        
        # Update b matrix using the Schur complement update formula
        b = b + np.outer(b[:, j0], (b[k, :] - b[i0, :])) / b[i0, j0]

        # Replace the j0-th row in the index set with i0
        ind[j0] = i0

        # Increment the iteration count
        iter += 1

    # If the loop finishes without convergence, return the current row indices
    ind = np.sort(ind)  # Sort the final indices for consistency
    return ind

def cross(T, ranks, n_iter=2, seed=None, rcond=1e-12):
    """
    TT-Cross with forward / backward sweeps.

    Parameters
    ----------
    T      : ndarray, the target tensor
    ranks  : list, TT ranks (len = d-1)
    n_iter : int, number of fwd-bwd iterations
    """
    rng = np.random.default_rng(seed)
    d   = T.ndim
    n   = T.shape
    r   = [1] + list(ranks) + [1]  

    #random init for right index sets J_k (k = 0 … d‑2)
    J = []
    for k in range(d - 1):
        tail   = n[k + 1:]
        J_k    = [tuple(rng.integers(0, tail[i]) for i in range(len(tail)))
                  for _ in range(r[k + 1])]
        J.append(J_k)

    # ================= main loop =================
    for _ in range(n_iter):
        # ---------- forward sweep ----------
        G, I_list = [], []
        I_prev = [()]                       # left index set for dim 0

        for k in range(d - 1):
            rp, nk, rk = len(I_prev), n[k], r[k + 1]

            # build sampling matrix  (rp*nk) × rk
            M = np.empty((rp * nk, rk))
            for a, left in enumerate(I_prev):
                for i in range(nk):
                    for j, right in enumerate(J[k]):
                        M[a * nk + i, j] = T[left + (i,) + right]

            rows = maxvol(M)                # pick rk rows
            C    = M[rows, :]
            try:
                U = np.linalg.solve(C.T, M.T).T
            except np.linalg.LinAlgError:   
                U = M @ np.linalg.pinv(C, rcond=rcond)

            G.append(U.reshape(rp, nk, -1))

            # update left index set
            I_prev = [I_prev[rw // nk] + (rw % nk,) for rw in rows]
            I_list.append(I_prev)

        # last core (no cross needed)
        r_last, nd = len(I_prev), n[-1]
        Gd = np.empty((r_last, nd, 1))
        for a, left in enumerate(I_prev):
            for j in range(nd):
                Gd[a, j, 0] = T[left + (j,)]
        G.append(Gd)

        if n_iter == 1:
            return G

        # ---------- backward sweep : refresh J ----------
        J_new, tail = [None] * (d - 1), [()]
        for k in range(d - 2, -1, -1):
            I_k, rk, nk1 = I_list[k], len(I_list[k]), n[k + 1]
            M = np.empty((rk, nk1 * len(tail)))
            for a, left in enumerate(I_k):
                for rid, right in enumerate(tail):
                    base = rid * nk1
                    for i in range(nk1):
                        M[a, base + i] = T[left + (i,) + right]

            cols = maxvol(M.T)
            J_k  = []
            for c in cols:
                rid, idx = divmod(c, nk1)
                J_k.append((idx,) + tail[rid])
            J_new[k], tail = J_k, J_k
        J = J_new

    return G