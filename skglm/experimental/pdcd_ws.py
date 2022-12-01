import warnings

import numpy as np
from numpy.linalg import norm

from numba import njit
from skglm.utils.jit_compilation import compiled_clone
from sklearn.exceptions import ConvergenceWarning


class PDCD_WS:
    """Primal-Dual Coordinate Descent solver with working sets.

    Solver inspired by [1] that uses working sets.

    Parameters
    ----------
    max_iter : int, optional
        The maximum number of iterations or equivalently the
        the maximum number solved subproblems.

    max_epochs : int, optional
        Maximum number of CD epochs on each subproblem.

    p0 : int, optional
        First working set size.

    tol : float, optional
        The tolerance for the optimization.

    dual_init : array, shape (n_samples,) default None
        The initialization of dual variables.
        If None, they are initialized as the 0 vector ``np.zeros(n_samples)``.

    return_p_objs : bool, default False
        If True, returns the values of the objective in each iteration.
        Otherwise returns an empty array.

    verbose : bool or int, default False
        Amount of verbosity. 0/False is silent.

    References
    ----------
    .. [1] Olivier Fercoq and Pascal Bianchi,
        "A Coordinate-Descent Primal-Dual Algorithm with Large Step Size and Possibly
        Nonseparable Functions", SIAM Journal on Optimization, 2020,
        https://epubs.siam.org/doi/10.1137/18M1168480,
        code: https://github.com/Badr-MOUFAD/Fercoq-Bianchi-solver

    .. [2] Mathurin Massias, Alexandre Gramfort, Joseph Salmon,
        "From safe screening rules to working sets for faster Lasso-type solvers",
        OPTML workshop at NIPS 2017, https://arxiv.org/abs/1703.07285v2
    """

    def __init__(self, max_iter=1000, max_epochs=1000, p0=100, tol=1e-6,
                 dual_init=None, return_p_objs=False, verbose=False):
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.dual_init = dual_init
        self.p0 = p0
        self.tol = tol
        self.verbose = verbose
        self.return_p_objs = return_p_objs

    def solve(self, X, y, datafit_, penalty_):
        datafit, penalty = PDCD_WS._validate_init(datafit_, penalty_)
        n_samples, n_features = X.shape

        # init steps
        dual_step = 1 / norm(X, ord=2)
        primal_steps = 1 / norm(X, axis=0, ord=2)

        # primal vars
        w = np.zeros(n_features)
        Xw = np.zeros(n_samples)

        # dual vars
        if self.dual_init is None:
            z = np.zeros(n_samples)
            z_bar = np.zeros(n_samples)
        else:
            z = self.dual_init.copy()
            z_bar = self.dual_init.copy()

        p_objs = []
        all_features = np.arange(n_features)

        for iter in range(self.max_iter):

            # check convergence
            opts_primal = penalty.subdiff_distance(w, X.T @ z, all_features)
            opt_dual = datafit.subdiff_distance(Xw, z, y)

            stop_crit = max(
                max(opts_primal),
                opt_dual
            )

            if self.verbose:
                current_p_obj = datafit.value(y, w, Xw) + penalty.value(w)
                print(
                    f"Iteration {iter+1}: {current_p_obj:.10f}, "
                    f"stopping crit: {stop_crit:.2e}")

            if self.return_p_objs:
                current_p_obj = datafit.value(y, w, Xw) + penalty.value(w)
                p_objs.append(current_p_obj)

            if stop_crit <= self.tol:
                break

            # build ws
            gsupp_size = (w != 0).sum()
            ws_size = max(min(self.p0, n_features),
                          min(n_features, 2 * gsupp_size))

            # similar to np.argsort()[-ws_size:] but without full sort
            ws = np.argpartition(opts_primal, -ws_size)[-ws_size:]

            # solve sub problem
            # inplace update of w, Xw, z, z_bar
            PDCD_WS._solve_subproblem(
                y, X, w, Xw, z, z_bar, datafit, penalty,
                primal_steps, dual_step, ws, self.max_epochs, tol_in=0.3*stop_crit)
        else:
            warnings.warn(
                f"PDCD_WS did not converge for tol={self.tol:.3e} "
                f"and max_iter={self.max_iter}.\n"
                "Considering increasing `max_iter` or decreasing `tol`.",
                category=ConvergenceWarning
            )

        return w, np.asarray(p_objs), stop_crit

    @staticmethod
    @njit
    def _solve_subproblem(y, X, w, Xw, z, z_bar, datafit, penalty,
                          primal_steps, dual_step, ws, max_epochs, tol_in):
        n_features = X.shape[1]
        past_pseudo_grad = np.zeros(len(ws))

        for epoch in range(max_epochs):

            for idx, j in enumerate(ws):
                # update primal
                old_w_j = w[j]
                past_pseudo_grad[idx] = X[:, j] @ (2 * z_bar - z)
                w[j] = penalty.prox_1d(
                    old_w_j - primal_steps[j] * past_pseudo_grad[idx],
                    primal_steps[j], j)

                # keep Xw syncr with X @ w
                delta_w_j = w[j] - old_w_j
                if delta_w_j:
                    Xw += delta_w_j * X[:, j]

                # update dual
                z_bar[:] = datafit.prox_conjugate(z + dual_step * Xw,
                                                  dual_step, y)
                z += (z_bar - z) / n_features

            # check convergence
            if epoch % 10 == 0:
                opts_primal_in = penalty.subdiff_distance(w, past_pseudo_grad, ws)
                opt_dual_in = datafit.subdiff_distance(Xw, z, y)

                stop_crit_in = max(
                    max(opts_primal_in),
                    opt_dual_in
                )

                if stop_crit_in <= tol_in:
                    break

    @staticmethod
    def _validate_init(datafit_, penalty_):
        # validate datafit
        missing_attrs = []
        for attr in ('prox_conjugate', 'subdiff_distance'):
            if not hasattr(datafit_, attr):
                missing_attrs.append(f"`{attr}`")

        if len(missing_attrs):
            raise AttributeError(
                "Datafit is not compatible with PDCD_WS solver.\n"
                "Datafit must implement `prox_conjugate` and `subdiff_distance`.\n"
                f"Missing {' and '.join(missing_attrs)}."
            )

        # jit compile classes
        compiled_datafit = compiled_clone(datafit_)
        compiled_penalty = compiled_clone(penalty_)

        return compiled_datafit, compiled_penalty
