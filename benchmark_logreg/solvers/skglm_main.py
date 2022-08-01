from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import warnings
    import numpy as np
    from skglm.estimators import SparseLogisticRegression
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):

    name = "skglm-main-branch"

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd
        n_samples = self.X.shape[0]

        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        self.logreg = SparseLogisticRegression(
            alpha=self.lmbd / n_samples, max_iter=1, max_epochs=50_000,
            tol=1e-12, fit_intercept=False, warm_start=False, verbose=False)

        # Cache Numba compilation
        self.run(1)

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros(self.X.shape[1])
        else:
            self.logreg.max_iter = n_iter
            self.logreg.fit(self.X, self.y)

            coef = self.logreg.coef_.flatten()
            self.coef = coef

    def get_result(self):
        return self.coef
