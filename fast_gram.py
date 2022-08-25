import time
import numpy as np
from numba import njit

from skglm.utils import make_correlated_data


@njit
def fast_gram_sparse(data, indices, indptr):
    # this needs indices to be sorted (sort_indices()), nasty bug otherwise
    n_features = len(indptr) - 1
    gram = np.zeros((n_features, n_features))

    for i in range(n_features):
        i_start, i_end = indptr[i], indptr[i + 1]
        gram[i, i] = (data[i_start:i_end] ** 2).sum()
        for j in range(i):
            j_start, j_end = indptr[j], indptr[j + 1]
            scal = 0
            i_idx = i_start
            j_idx = j_start
            while i_idx < i_end and j_idx < j_end:
                if indices[i_idx] < indices[j_idx]:
                    i_idx += 1
                elif indices[i_idx] > indices[j_idx]:
                    j_idx += 1
                else:  # they match
                    scal += data[i_idx] * data[j_idx]
                    i_idx += 1
                    j_idx += 1
            gram[i, j] = scal
            gram[j, i] = scal
    return gram


X, _, _ = make_correlated_data(10, 5, random_state=0, X_density=0.5)
fast_gram_sparse(X.data, X.indices, X.indptr)

X, _, _ = make_correlated_data(1000, 200, random_state=0, X_density=0.5)
t0 = time.time()
gram2 = fast_gram_sparse(X.data, X.indices, X.indptr)
print(f"us: {time.time() -t0:.3f} s")

t0 = time.time()
gram1 = (X.T @ X).toarray()
print(f"sp: {time.time() -t0:.3f} s")


print(np.linalg.norm(gram1 - gram2))
