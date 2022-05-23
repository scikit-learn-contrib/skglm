import numpy as np
from numpy.testing import assert_allclose

from skglm.penalties.block_separable import SparseGroupL1
from skglm.penalties.separable import L1
from skglm.utils import grp_converter


def test_equivalence_SparseGroupLasso_L1():
    n_features, alpha = 1000, 1.
    grp_ptr, grp_indices = grp_converter(1, n_features)
    weights = np.array([1 for _ in range(n_features)], dtype=np.float64)

    l1_penalty = L1(alpha)

    lasso_penalty = SparseGroupL1(
        alpha=alpha, tau=1.,
        grp_ptr=grp_ptr, grp_indices=grp_indices,
        weights=weights
    )

    single_group_penalty = SparseGroupL1(
        alpha=alpha, tau=0.,
        grp_ptr=grp_ptr, grp_indices=grp_indices,
        weights=weights
    )

    penalties = [l1_penalty, lasso_penalty, single_group_penalty]

    rnd = np.random.RandomState(42)
    w = rnd.normal(loc=5, scale=4, size=n_features)

    values = np.array([penalty.value(w)
                       for penalty in penalties])

    proxs = np.array([
        l1_penalty.prox_1d(w[0], 1., 0),
        lasso_penalty.prox_1feat(+w[0:1], 1., 0)[0],
        single_group_penalty.prox_1feat(+w[0:1], 1., 0)[0]
    ])

    assert_allclose(values.std(), 0)
    assert_allclose(proxs.std(), 0)


if __name__ == '__main__':
    test_equivalence_SparseGroupLasso_L1()
