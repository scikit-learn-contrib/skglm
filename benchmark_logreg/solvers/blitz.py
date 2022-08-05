from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    import blitzl1


class Solver(BaseSolver):

    name = 'blitz'

    # 'pip:git+https://github.com/tbjohns/blitzl1.git@master'

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

        blitzl1.set_use_intercept(False)
        blitzl1.set_tolerance(1e-9)
        self.problem = blitzl1.LogRegProblem(self.X, self.y)

    def run(self, n_iter):
        self.coef_ = self.problem.solve(self.lmbd, max_iter=n_iter).x

    def get_result(self):
        return self.coef_.flatten()
