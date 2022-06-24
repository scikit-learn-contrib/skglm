from benchopt import BaseObjective
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm

    from skglm.datafits import Quadratic
    from skglm.penalties import L1


class Objective(BaseObjective):
    name = "Lasso objective"

    parameters = {
        'rho': [1., 1e-1, 1e-2, 1e-3],
    }

    def __init__(self, rho):
        self.rho = rho

    def set_data(self, X, y):
        self.X, self.y = X, y
        alpha_max = norm(X.T @ y, ord=np.inf) / len(y)
        self.alpha = self.rho * alpha_max

        self.datafit = Quadratic()
        self.penalty = L1(self.alpha)

    def compute(self, beta):
        return (self.datafit.value(self.y, beta, self.X @ beta)
                + self.penalty.value(beta))

    def to_dict(self):
        return dict(X=self.X, y=self.y, alpha=self.alpha)
