# -----------------------------------------------------------------------------
# A toy example for 2D matrix cross approximation
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



"""
Cross approximation for matrices
"""
# Parameters and data
n_max = 500  # Maximum number of iterations
rank = 4  # Rank
nbStates = 4  # Number of Gaussians
Mu = np.array([[2, 4, 7, 9], [4, 9, 2, 6]])
sigma = np.array([1, 5, 1, 1])

# Generate reference distribution as a GMM
nbVar = [10, 10]
A = np.zeros(nbVar)

for i in range(nbVar[0]):
    for j in range(nbVar[1]):
        for k in range(nbStates):
            eTmp = np.array([i, j]) - Mu[:, k]
            A[i, j] += np.exp(-eTmp.T @ eTmp / sigma[k]) / nbStates

A += np.random.rand(*nbVar) * 1E-2

# Cross matrix approximation
iv = np.random.randint(nbVar[1], size=rank)  # Random initial column list
A_estold = np.zeros(nbVar)

iu = np.random.randint(nbVar[0], size=rank)  # Random initial row list

for n in range(n_max):
    if n % 2 == 0:
        # QR decomposition to update columns
        V, _ = qr(A[iu, :].T)  # QR decomposition on rows
        iv = maxvol(V[:, :rank])  # Find maximal volume submatrix columns
    else:
        # QR decomposition to update rows
        U, _ = qr(A[:, iv])  # QR decomposition on columns
        iu = maxvol(U[:, :rank])  # Find maximal volume submatrix rows

    # Reconstruction
    A_iu_iv = A[np.ix_(iu, iv)]  # Get the submatrix A[iu, iv]
    A_est = A[:, iv] @ np.linalg.lstsq(A_iu_iv, A[iu, :], rcond=None)[0]

    # Convergence check
    if np.linalg.norm(A_est - A_estold) < 1E-2:
        break

    A_estold = A_est

print(f"Cross matrix approximation converged in {n+1} iterations.")

# Plots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].set_title('Original')
axes[0].imshow(A, aspect='equal', origin='upper')
axes[0].axis('off')

axes[1].set_title('Reconstructed')
axes[1].imshow(A_est, aspect='equal', origin='upper')
axes[1].axis('off')

plt.show()

