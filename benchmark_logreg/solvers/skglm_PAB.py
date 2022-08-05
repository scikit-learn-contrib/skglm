from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from skglm.datafits import Logistic
    from skglm.utils import compiled_clone
    from skglm.penalties import L1
    from skglm.prototype_PN.pn_PAB import prox_newton_solver


class Solver(BaseSolver):

    name = "skglm-PN-PAB"

    def __init__(self):
        pass

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd
        n_samples = self.X.shape[0]

        self.log_datafit = compiled_clone(Logistic())
        self.l1_penalty = compiled_clone(L1(self.lmbd / n_samples))

        self.tol = 1e-9 * n_samples  # scale tol

        # Cache Numba compilation
        self.run(5)

    def run(self, n_iter):
        self.coef = prox_newton_solver(self.X, self.y, self.log_datafit,
                                       self.l1_penalty, tol=self.tol,
                                       max_iter=n_iter)[0]

    def get_result(self):
        return self.coef
