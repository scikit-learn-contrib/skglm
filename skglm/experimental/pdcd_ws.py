import warnings

import numpy as np
from numpy.linalg import norm

from numba import njit
from skglm.utils.jit_compilation import compiled_clone
from sklearn.exceptions import ConvergenceWarning


class PDCD_WS:
    r"""Primal-Dual Coordinate Descent solver with working sets.

    It solves::

        \min_w F(Xw) + G(w) \Leftrightarrow \min_w \max_z <Xw, z> + G(w) - F^*(z)

    where :math:`F` is the datafit term (:math:`F^*` it's Fenchel conjugate)
    and :math:`G` is the penalty term.

    The datafit is required to be convex and proximable. Also, the penalty
    is also required to be convex, separable, and proximable.

    The solver is inspired by [1] and uses working sets [2].
    The working sets are built using a fixed point distance strategy
    where each feature is assigned a score based how much it maps
    to itself when performing a primal update::

        \text{score}_j = \abs{w_j - prox_{G_j}(w_j - \tau_j <X_j, z>)}

    where :maths:`\tau_j` is the primal step associated with the j-th feature.

    Parameters
    ----------
    max_iter : int, optional
        The maximum number of iterations or equivalently the
        the maximum number of solved subproblems.

    max_epochs : int, optional
        Maximum number of primal CD epochs on each subproblem.

    p0 : int, optional
        First working set size.

    tol : float, optional
        The tolerance for the optimization.

    verbose : bool or int, default False
        Amount of verbosity. 0/False is silent.

    References
    ----------
    .. [1] Olivier Fercoq and Pascal Bianchi,
        "A Coordinate-Descent Primal-Dual Algorithm with Large Step Size and Possibly
        Nonseparable Functions", SIAM Journal on Optimization, 2020,
        https://epubs.siam.org/doi/10.1137/18M1168480,
        code: https://github.com/Badr-MOUFAD/Fercoq-Bianchi-solver

    .. [2] Bertrand, Q. and Klopfenstein, Q. and Bannier, P.-A. and Gidel, G.
           and Massias, M.
           "Beyond L1: Faster and Better Sparse Models with skglm", 2022
           https://arxiv.org/abs/2204.07826
    """

    def __init__(self, max_iter=1000, max_epochs=1000,
                 p0=100, tol=1e-6, verbose=False):
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.tol = tol
        self.verbose = verbose

    def solve(self, X, y, datafit_, penalty_, w_init=None, Xw_init=None):
        datafit, penalty = PDCD_WS._validate_init(datafit_, penalty_)
        n_samples, n_features = X.shape

        # init steps
        # Despite violating the conditions mentioned in [1]
        # this choice of steps yield in practice a convergent algorithm
        # with better speed of convergence
        dual_step = 1 / norm(X, ord=2)
        primal_steps = 1 / norm(X, axis=0, ord=2)

        # primal vars
        w = np.zeros(n_features) if w_init is None else w_init
        Xw = np.zeros(n_samples) if Xw_init is None else Xw_init

        # dual vars
        z = y.copy()
        z_bar = y.copy()

        p_objs = []
        stop_crit = 0.
        all_features = np.arange(n_features)

        for iteration in range(self.max_iter):

            # check convergence
            opts_primal = _scores_primal(
                X, w, z, penalty, primal_steps, all_features)

            opt_dual = _score_dual(
                y, z, Xw, datafit, dual_step)

            stop_crit = max(
                max(opts_primal),
                opt_dual
            )

            if self.verbose:
                current_p_obj = datafit.value(y, w, Xw) + penalty.value(w)
                print(
                    f"Iteration {iteration+1}: {current_p_obj:.10f}, "
                    f"stopping crit: {stop_crit:.2e}")

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

            current_p_obj = datafit.value(y, w, Xw) + penalty.value(w)
            p_objs.append(current_p_obj)
        else:
            warnings.warn(
                f"PDCD_WS did not converge for tol={self.tol:.3e} "
                f"and max_iter={self.max_iter}.\n"
                "Considering increasing `max_iter` or `tol`.",
                category=ConvergenceWarning
            )

        return w, np.asarray(p_objs), stop_crit

    @staticmethod
    @njit
    def _solve_subproblem(y, X, w, Xw, z, z_bar, datafit, penalty,
                          primal_steps, dual_step, ws, max_epochs, tol_in):
        n_features = X.shape[1]

        for epoch in range(max_epochs):

            for j in ws:
                # update primal
                old_w_j = w[j]
                pseudo_grad = X[:, j] @ (2 * z_bar - z)
                w[j] = penalty.prox_1d(
                    old_w_j - primal_steps[j] * pseudo_grad,
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
                opts_primal_in = _scores_primal(
                    X, w, z, penalty, primal_steps, ws)

                opt_dual_in = _score_dual(
                    y, z, Xw, datafit, dual_step)

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


@njit
def _scores_primal(X, w, z, penalty, primal_steps, ws):
    scores_ws = np.zeros(len(ws))

    for idx, j in enumerate(ws):
        next_w_j = penalty.prox_1d(w[j] - primal_steps[j] * X[:, j] @ z,
                                   primal_steps[j], j)
        scores_ws[idx] = abs(w[j] - next_w_j)

    return scores_ws


@njit
def _score_dual(y, z, Xw, datafit, dual_step):
    next_z = datafit.prox_conjugate(z + dual_step * Xw,
                                    dual_step, y)
    return norm(z - next_z, ord=np.inf)
