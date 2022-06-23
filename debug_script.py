r"""Example with:

u_{n+1} = \begin{cases}
            u_{n}            & if n \bmod 3
            \rho u_{n} + b   & otherwise
          \end{cases}
"""
import numpy as np
from numpy.linalg import norm

from debug_script_utils import AA_singular, AA_usual


def assess_AA(aa_class):
    max_iter, tol = 1000, 1e-9
    n_features = 5
    X = np.eye(n_features)

    b = 1
    np.random.seed(0)
    rho = np.random.rand(n_features)
    w_star = b / (1 - rho)

    accelerator = aa_class(K=5)
    w = np.ones(n_features)
    Xw = X @ w
    for i in range(max_iter):
        w, Xw = accelerator.extrapolate(w, Xw)
        if i % 3:
            w = rho * w + b
            Xw = X @ w

        stop_crit = norm(w - w_star, ord=np.inf)
        if stop_crit < tol:
            break

    return stop_crit, i


if __name__ == '__main__':
    stop_crit, n_iter = assess_AA(AA_usual)
    print(f"usual AA: {stop_crit} , iter {n_iter}")

    print("********************")
    print("********************")

    stop_crit, n_iter = assess_AA(AA_singular)
    print(f"singular AA : {stop_crit} , iter {n_iter}")
