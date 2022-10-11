import numpy as np
from numpy.linalg import norm
from skglm.estimators import Lasso


class ReweightedLasso(Lasso):
    def __init__(self, alpha=1., n_reweight=5, reweight_penalty=None, *args, **kwargs):
        super().__init__(alpha=alpha, *args, **kwargs)
        self.n_reweight = n_reweight
        self.reweight_penalty = reweight_penalty or self._reweight_penalty
        self.loss_history_ = None

    def fit(self, X, y):
        self.loss_history_ = []
        n_samples, n_features = X.shape
        weights = np.ones(n_features)
        # XXX: dot product X @ w is slow in high-dimension, to be improved
        objective = (lambda w: np.sum((y - X @ w) ** 2) / (2 * n_samples)
                     + self.alpha * np.sqrt(norm(w)))

        for l in range(self.n_reweight):
            # trick: rescaling the weights (XXX: sparse X would become dense?)
            scaled_X = X / weights
            super().fit(scaled_X, y)
            scaled_coef = self.coef_ / weights

            # updating the weights
            weights = self.reweight_penalty(scaled_coef)

            loss = objective(scaled_coef)
            self.loss_history_.append(loss)

            if self.verbose:
                print("#" * 10)
                print(f"[REWEIGHT] iteration {l} :: loss {loss}")
                print("#" * 10)

        self.coef_ = scaled_coef

    @staticmethod
    def _reweight_penalty(coef):
        nrm = np.sqrt(norm(coef))
        return 1 / (2 * nrm + np.finfo(float).eps)

