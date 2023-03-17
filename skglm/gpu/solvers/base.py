from abc import abstractmethod

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spicy_linalg


class BaseFistaSolver:

    @abstractmethod
    def solve(self, X, y, lmbd):
        ...

    @staticmethod
    def get_lipschitz_cst(X):
        if sparse.issparse(X):
            return spicy_linalg.svds(X, k=1)[1][0] ** 2

        return np.linalg.norm(X, ord=2) ** 2
