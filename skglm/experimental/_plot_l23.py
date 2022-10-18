import dataclasses as dc

import numpy as np
import matplotlib.pylab as plt

@dc.dataclass
class LASSOProblem:
    A: np.ndarray = None
    b: np.ndarray = None
    lam: float = 0.

    @classmethod
    def random_problem(cls, rows=500, cols=1000) -> 'LASSOProblem':
        return cls(
            A=np.random.randn(rows, cols),
            b=np.random.randn(rows),
            lam=1.
        )


lasso_problem = LASSOProblem.random_problem()


import scipy.optimize

def LASSO_reparametrized_nonlinear_least_squares(lasso: LASSOProblem) -> np.ndarray:
    def func(y):
        return np.hstack((
            lasso.A.dot(y**3) - lasso.b,
            np.sqrt(lasso.lam) * y
        ))

    def dfunc(y):
        return np.vstack((
          3 * lasso.A * y[None, :]**2,
          np.sqrt(lasso.lam) * np.eye(len(y))
        ))


    x0 = np.linalg.lstsq(lasso.A, lasso.b)[0]
    res_scipy = scipy.optimize.least_squares(
      func, x0, jac=dfunc, method='lm',
      max_nfev=100,
    )

    return res_scipy

res_scipy = LASSO_reparametrized_nonlinear_least_squares(lasso_problem)
x_reparametrized_nonlinear_least_squares = res_scipy.x**3

import skglm

def LASSO_skglm(lasso: LASSOProblem) -> np.ndarray:
    model = skglm.GeneralizedLinearEstimator(
        datafit=skglm.datafits.Quadratic(),
        penalty=skglm.penalties.L2_3(lasso.lam / len(lasso.b)),
        solver=skglm.solvers.AndersonCD(
            tol=1e-12, verbose=True, fit_intercept=False,
            ws_strategy="fixpoint"
        ),
    )

    model.fit(lasso.A, lasso.b)

    return model.coef_

x_skglm = LASSO_skglm(lasso_problem)


def objectif_function(A, b, x, lam):
    return ((A @ x - b) ** 2).sum() / 2 + skglm.penalties.L2_3(lam).value(x)

obj_reparam = objectif_function(
    lasso_problem.A, lasso_problem.b, x_reparametrized_nonlinear_least_squares, lasso_problem.lam)

obj_skglm = objectif_function(
    lasso_problem.A, lasso_problem.b, x_skglm, lasso_problem.lam)

print("Value of the objective function for the reparametrized algorithm: %.1f" % obj_reparam)
print("Value of the objective function for skglm: %.1f" % obj_skglm)
