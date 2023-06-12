from skglm.solvers import BaseSolver


class BFGS(BaseSolver):
    """A wrapper for scipy BFGS solver."""

    def __init__(self, max_iter=50, tol=1e-4):
        self.max_iter = max_iter
        self.tol = tol

    def solve(X, y, datafit, penalty, w_init=None, Xw_init=None):
        return
