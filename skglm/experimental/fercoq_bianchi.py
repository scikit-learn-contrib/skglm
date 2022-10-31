import numpy as np
from numpy.linalg import norm

from numba import njit


def fercoq_bianchi(X, y, datafit, penalty, max_iter=1000, tol=1e-4,
                   verbose=0, random_state=0):
    """Primal-dual CD algorithm for problems with non-smooth datafits.

    Adaptation of Fercoq & Bianchi primal-dual CD algorithm [1].

    Parameters
    ----------
        X : array, shape (n_samples, n_features)
            Training data.

        y : array, shape (n_samples,)
            Target values.

        datafit : instance of Datafit class
            Datafitting term.

        penalty : instance of Penalty class
            Penalty used in the model.

        max_iter : int, default 1000
            Maximum number of iterations.

        tol : float, default 1e-4
            Tolerance for convergence.

        verbose : bool, default False
            Amount of verbosity. 0/False is silent.

        random_state : int, default 0
            The seed to control the randomly selected features to be updated
            at each iteration.

    Returns
    -------
        w : array, shape (n_features,)
            Regression coefficients.

        p_objs_out : array, shape (n_iter,)
            The objective values at every outer iteration.

        stop_crit : float
            Value of stopping criterion at convergence.

    References
    ----------
    .. [1] Olivier Fercoq and Pascal Bianchi
        "A Coordinate-Descent Primal-Dual Algorithm with Large Step Size and Possibly
        Nonseparable Functions", SIAM Journal on Optimization, 2020,
        https://epubs.siam.org/doi/10.1137/18M1168480
    """
    n_samples, n_features = X.shape
    rng = np.random.RandomState(random_state)
    all_features = np.arange(n_features)

    stop_crit = 0.
    p_objs_out = []

    # primal variable
    w = np.zeros(n_features)
    Xw = np.zeros(n_samples)

    # dual variable
    z = np.zeros(n_samples)

    # constants verifies: tau_j < 1 / (||X|| * (2n-1) * sigma * ||X_j||^2)
    sigma = 1 / ((2 * n_features - 1) * norm(X, ord=2))
    tau = 1 / (X ** 2).sum(axis=0)  # 1 / squared norm of columns

    for iter in range(max_iter):

        # random CD with inplace update of w, Xw, and z
        selected_features = rng.choice(n_features, n_features)
        _primal_dual_cd_epoch(y, X, w, Xw, z, datafit, penalty,
                              selected_features, sigma, tau)

        # check convergence (so far only the primal optimally condition)
        if iter % 10 == 0:
            otp = penalty.subdiff_distance(w, X.T @ z, all_features)
            stop_crit = np.max(otp)

            if verbose:
                p_obj = datafit.value(y, w, Xw) + penalty.value(w)
                print(
                    f"Iteration {iter+1}: {p_obj:.10f}, "
                    f"stopping crit: {stop_crit:.2e}"
                )

            if stop_crit <= tol:
                break

        p_obj = datafit.value(y, w, Xw) + penalty.value(w)
        p_objs_out.append(p_obj)
    return w, np.asarray(p_objs_out), stop_crit


@njit
def _primal_dual_cd_epoch(y, X, w, Xw, z, datafit, penalty,
                          features, sigma, tau):
    # inplace update of w, Xw
    n_features = X.shape[1]

    for j in features:
        # dual update
        z_bar = datafit.prox_conjugate(y, X, z + sigma * Xw, sigma)
        z += (z_bar - z) / n_features

        # primal update
        old_w_j = w[j]
        w[j] = penalty.prox_1d(old_w_j - tau[j] * (X[:, j] @ (2 * z_bar - z)),
                               tau[j], j)

        # keep Xw synchronized with X @ W
        delta_w_j = w[j] - old_w_j
        if delta_w_j != 0:
            Xw += delta_w_j * X[:, j]


if __name__ == '__main__':
    pass
