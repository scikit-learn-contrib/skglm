import numpy as np
import scipy.sparse

from skglm.utils.sparse_ops import spectral_norm, sparse_columns_slice


def test_spectral_norm():
    n_samples, n_features = 50, 60
    A_sparse = scipy.sparse.random(
        n_samples, n_features, density=0.7, format='csc', random_state=37)

    A_bundles = (A_sparse.data, A_sparse.indptr, A_sparse.indices)
    spectral_norm_our = spectral_norm(*A_bundles, n_samples=A_sparse.shape[0])
    spectral_norm_sp = scipy.sparse.linalg.svds(A_sparse, k=1)[1]

    np.testing.assert_allclose(spectral_norm_our, spectral_norm_sp)


def test_slice_cols_sparse():
    n_samples, n_features = 20, 50
    rng = np.random.RandomState(546)

    M = scipy.sparse.random(
        n_samples, n_features, density=0.9, format="csc", random_state=rng)
    cols = rng.choice(n_features, size=n_features // 10, replace=False)

    sub_M_data, sub_M_indptr, sub_M_indices = sparse_columns_slice(
        cols, M.data, M.indptr, M.indices)
    sub_M = scipy.sparse.csc_matrix(
        (sub_M_data, sub_M_indices, sub_M_indptr), shape=(n_samples, len(cols)))

    np.testing.assert_array_equal(sub_M.toarray(), M.toarray()[:, cols])


if __name__ == "__main__":
    pass
