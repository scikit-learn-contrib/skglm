from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    from skglm.datafits import Quadratic
    from skglm.penalties import L1
    from skglm.solvers.cd_solver import cd_solver


class Solver(BaseSolver):
    name = "cd_solver - AA class"

    def __init__(self) -> None:
        pass

    def set_objective(self, X, y, alpha):
        self.X, self.y = X, y
        self.alpha = alpha

        # init solver
        self.w = np.zeros(X.shape[1])
        self.Xw = np.zeros(X.shape[0])

        self.datafit = Quadratic()
        self.penalty = L1(alpha)

        self.run(n_iter=10)  # cache numba compilation

    def run(self, n_iter):
        X, y = self.X, self.y
        datafit, penalty = self.datafit, self.penalty

        self.w = cd_solver(X, y, datafit, penalty, w=self.w, Xw=self.Xw,
                           max_iter=n_iter, tol=1e-12)[0]

    def get_result(self):
        return self.w
