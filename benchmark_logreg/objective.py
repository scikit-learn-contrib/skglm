import numpy as np
from benchopt import BaseObjective


class Objective(BaseObjective):
    name = "Sparse Logistic Regression"

    parameters = {
        'fit_intercept': [False],
        'reg': [1., .5, .1, .05]
    }

    def __init__(self, reg=.1, fit_intercept=False):
        self.reg = reg
        self.fit_intercept = fit_intercept

    def set_data(self, X, y):
        if set(y) != set([-1, 1]):
            raise ValueError(
                f"y must contain only -1 or 1 as values. Got {set(y)}"
            )
        self.X, self.y = X, y
        self.lmbd = self.reg * self._get_lambda_max()

    def get_one_solution(self):
        n_features = self.X.shape[1]
        if self.fit_intercept:
            n_features += 1
        return np.zeros(n_features)

    def compute(self, beta):
        beta = beta.flatten().astype(np.float64)
        y_X_beta = self.y * (self.X @ beta)
        l1 = abs(beta).sum()
        return np.log(1 + np.exp(-y_X_beta)).sum() + self.lmbd * l1

    def _get_lambda_max(self):
        return abs(self.X.T @ self.y).max() / 2

    def to_dict(self):
        return dict(X=self.X, y=self.y, lmbd=self.lmbd)
