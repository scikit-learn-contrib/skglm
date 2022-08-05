import warnings

from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from celer import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):

    name = 'celer'

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        self.clf = LogisticRegression(
            penalty='l1', C=1/self.lmbd, max_iter=1,
            max_epochs=100000, p0=10, verbose=False, tol=1e-9,
            fit_intercept=False
        )

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        return self.clf.coef_.flatten()
