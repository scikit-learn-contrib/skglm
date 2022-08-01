import warnings
from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression


class Solver(BaseSolver):

    name = 'sklearn'

    parameters = {
        'solver': ['liblinear'],
    }

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        self.clf = LogisticRegression(
            solver=self.solver, C=1 / self.lmbd,
            penalty='l1', fit_intercept=False,
            tol=1e-12)

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        return self.clf.coef_.flatten()
