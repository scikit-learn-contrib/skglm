import numpy as np
from numpy.linalg import norm
from numba import njit


@njit
def spectral_norm(X_data, X_indptr, X_indices, n_samples,
                  max_iter=100, tol=1e-6):
    """Compute the spectral norm of sparse matrix ``X`` with power method.

    Parameters
    ----------
    X_data : array, shape (n_elements,)
         ``data`` attribute of the sparse CSC matrix ``X``.

    X_indptr : array, shape (n_features + 1,)
         ``indptr`` attribute of the sparse CSC matrix ``X``.

    X_indices : array, shape (n_elements,)
         ``indices`` attribute of the sparse CSC matrix ``X``.

    n_samples : int
        number of rows of ``X``.

    max_iter : int, default 20
        Maximum number of power method iterations.

    tol : float, default 1e-6
        Tolerance for convergence.

    Returns
    -------
    eigenvalue : float
        The largest singular value of ``X``.

    References
    ----------
    .. [1] Alfio Quarteroni, Riccardo Sacco, Fausto Saleri "Numerical Mathematics",
        chapter 5, page 192-195.
    """
    # init vec with norm(vec) == 1.
    eigenvector = np.random.randn(n_samples)
    eigenvector /= norm(eigenvector)
    eigenvalue = 1.

    for _ in range(max_iter):
        vec = _XXT_dot_vec(X_data, X_indptr, X_indices, eigenvector, n_samples)
        norm_vec = norm(vec)
        eigenvalue = vec @ eigenvector

        # norm(X @ X.T @ eigenvector - eigenvalue * eigenvector) <= tol
        # inequality (5.25) in ref [1] is squared
        if norm_vec ** 2 - eigenvalue ** 2 <= tol ** 2:
            break

        eigenvector = vec / norm_vec

    return np.sqrt(eigenvalue)


@njit
def sparse_columns_slice(cols, X_data, X_indptr, X_indices):
    """Select a sub matrix from CSC sparse matrix.

    Similar to ``X[:, cols]`` but for ``X`` a CSC sparse matrix.

    Parameters
    ----------
    cols : array of int
        Columns to select in matrix ``X``.

    X_data : array, shape (n_elements,)
        ``data`` attribute of the sparse CSC matrix ``X``.

    X_indptr : array, shape (n_features + 1,)
        ``indptr`` attribute of the sparse CSC matrix ``X``.

    X_indices : array, shape (n_elements,)
        ``indices`` attribute of the sparse CSC matrix ``X``.

    Returns
    -------
    sub_X_data, sub_X_indptr, sub_X_indices
        The ``data``, ``indptr``, and ``indices`` attributes of the sub matrix.
    """
    nnz = sum([X_indptr[j+1] - X_indptr[j] for j in cols])

    sub_X_indptr = np.zeros(len(cols) + 1, dtype=cols.dtype)
    sub_X_indices = np.zeros(nnz, dtype=X_indices.dtype)
    sub_X_data = np.zeros(nnz, dtype=X_data.dtype)

    for idx, j in enumerate(cols):
        n_elements = X_indptr[j+1] - X_indptr[j]
        sub_X_indptr[idx + 1] = sub_X_indptr[idx] + n_elements

        col_j_slice = slice(X_indptr[j], X_indptr[j+1])
        col_idx_slice = slice(sub_X_indptr[idx], sub_X_indptr[idx+1])

        sub_X_indices[col_idx_slice] = X_indices[col_j_slice]
        sub_X_data[col_idx_slice] = X_data[col_j_slice]

    return sub_X_data, sub_X_indptr, sub_X_indices


@njit
def _XXT_dot_vec(X_data, X_indptr, X_indices, vec, n_samples):
    # computes X @ X.T @ vec, with X csc encoded
    return _X_dot_vec(X_data, X_indptr, X_indices,
                      _XT_dot_vec(X_data, X_indptr, X_indices, vec), n_samples)


@njit
def _X_dot_vec(X_data, X_indptr, X_indices, vec, n_samples):
    # compute X @ vec, with X csc encoded
    result = np.zeros(n_samples)

    # loop over features
    for j in range(len(X_indptr) - 1):
        if vec[j] == 0:
            continue

        col_j_rows_idx = slice(X_indptr[j], X_indptr[j+1])
        result[X_indices[col_j_rows_idx]] += vec[j] * X_data[col_j_rows_idx]

    return result


@njit
def _XT_dot_vec(X_data, X_indptr, X_indices, vec):
    # compute X.T @ vec, with X csc encoded
    n_features = len(X_indptr) - 1
    result = np.zeros(n_features)

    for j in range(n_features):
        for idx in range(X_indptr[j], X_indptr[j+1]):
            result[j] += X_data[idx] * vec[X_indices[idx]]

    return result


@njit(fastmath=True)
def _sparse_xj_dot(X_data, X_indptr, X_indices, j, other):
    # Compute X[:, j] @ other in case X sparse
    res = 0.
    for i in range(X_indptr[j], X_indptr[j+1]):
        res += X_data[i] * other[X_indices[i]]
    return res
