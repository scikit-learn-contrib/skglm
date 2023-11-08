import numpy as np
from numpy.linalg import norm
from numba import float64

from skglm.datafits.base import BaseMultitaskDatafit


class QuadraticMultiTask(BaseMultitaskDatafit):
    """Quadratic datafit used for multi-task regression.

    The datafit reads:

    .. math: 1 / (2 xx n_"samples") ||Y - XW||_F ^ 2

    Attributes
    ----------
    XtY : array, shape (n_features, n_tasks)
        Pre-computed quantity used during the gradient evaluation.
    """

    def __init__(self):
        pass

    def get_spec(self):
        spec = (
            ('XtY', float64[:, :]),
        )
        return spec

    def params_to_dict(self):
        return dict()

    def get_lipschitz(self, X, Y):
        n_samples, n_features = X.shape

        lipschitz = np.zeros(n_features)
        for j in range(n_features):
            lipschitz[j] = norm(X[:, j]) ** 2 / n_samples

        return lipschitz

    def get_lipschitz_sparse(self, X_data, X_indptr, X_indices, Y):
        n_samples, n_tasks = Y.shape
        n_features = len(X_indptr) - 1

        lipschitz = np.zeros(n_features)
        for j in range(n_features):
            nrm2 = 0.
            for idx in range(X_indptr[j], X_indptr[j + 1]):
                nrm2 += X_data[idx] ** 2

            lipschitz[j] = nrm2 / n_samples

        return lipschitz

    def initialize(self, X, Y):
        """Compute optimization quantities before fitting on X and Y."""
        self.XtY = X.T @ Y

    def initialize_sparse(self, X_data, X_indptr, X_indices, Y):
        """Pre-computations before fitting on X and Y, when X is sparse."""
        _, n_tasks = Y.shape
        n_features = len(X_indptr) - 1

        self.XtY = np.zeros((n_features, n_tasks))
        for j in range(n_features):
            xtY = np.zeros(n_tasks)
            for idx in range(X_indptr[j], X_indptr[j + 1]):
                for t in range(n_tasks):
                    xtY[t] += X_data[idx] * Y[X_indices[idx], t]

            self.XtY[j, :] = xtY

    def value(self, Y, W, XW):
        """Value of datafit at matrix W."""
        n_samples = Y.shape[0]
        return np.sum((Y - XW) ** 2) / (2 * n_samples)

    def gradient_j(self, X, Y, W, XW, j):
        """Gradient with respect to j-th coordinate of W."""
        n_samples = X.shape[0]
        return (X[:, j] @ XW - self.XtY[j, :]) / n_samples

    def gradient_j_sparse(self, X_data, X_indptr, X_indices, Y, XW, j):
        """Gradient with respect to j-th coordinate of W when X is sparse."""
        n_samples, n_tasks = Y.shape
        XjTXW = np.zeros(n_tasks)
        for t in range(n_tasks):
            for i in range(X_indptr[j], X_indptr[j+1]):
                XjTXW[t] += X_data[i] * XW[X_indices[i], t]
        return (XjTXW - self.XtY[j, :]) / n_samples

    def full_grad_sparse(self, X_data, X_indptr, X_indices, Y, XW):
        """Compute the full gradient when X is sparse."""
        n_features = X_indptr.shape[0] - 1
        n_samples, n_tasks = Y.shape
        grad = np.zeros((n_features, n_tasks))
        for j in range(n_features):
            XjTXW = np.zeros(n_tasks)
            for t in range(n_tasks):
                for i in range(X_indptr[j], X_indptr[j+1]):
                    XjTXW[t] += X_data[i] * XW[X_indices[i], t]
            grad[j, :] = (XjTXW - self.XtY[j, :]) / n_samples
        return grad

    def intercept_update_step(self, Y, XW):
        return np.sum(XW - Y, axis=0) / len(Y)
