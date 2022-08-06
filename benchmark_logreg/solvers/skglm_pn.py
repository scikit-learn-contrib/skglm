from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from skglm.prototype_PN.log_datafit import Pr_LogisticRegression
    from skglm.prototype_PN.pn_solver import pn_solver
    from skglm.penalties import L1
    from skglm.utils import compiled_clone


class Solver(BaseSolver):

    name = "skglm-PN"

    parameters = {
        'use_acc': [True, False]
    }

    def __init__(self, use_acc):
        self.use_acc = use_acc

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd
        n_samples = self.X.shape[0]

        self.log_datafit = compiled_clone(Pr_LogisticRegression())
        self.l1_penalty = compiled_clone(L1(self.lmbd / n_samples))

        self.tol = 1e-9  # scale tol

        # Cache Numba compilation
        self.run(5)

    def run(self, n_iter):
        self.coef = pn_solver(self.X, self.y, self.log_datafit,
                              self.l1_penalty, tol=self.tol,
                              use_acc=self.use_acc, max_iter=n_iter)[0]

    def get_result(self):
        return self.coef
