import warnings
import numpy as np
from numba import njit
from scipy.sparse import issparse

from skglm.solvers.base import BaseSolver
from skglm.utils.anderson import AndersonAcceleration


class GramCD(BaseSolver):
    r"""Coordinate descent solver keeping the gradients up-to-date with Gram updates.

    This solver should be used when ``n_features`` < ``n_samples``, and computes the
    (``n_features``, ``n_features``) Gram matrix which comes with an overhead. It is
    only suited to Quadratic datafits.

    It minimizes:

    .. math:: 1 / (2 xx n_"samples") ||y - Xw||^2 + "penalty"(w)

    which can be rewritten as:

    .. math:: 1 / (2 xx n_"samples") w^T Q w - 1 / n_"samples" q^T w + "penalty"(w)

    where:

    .. math:: Q = X^T X " (gram matrix),  and " q = X^T y

    Attributes
    ----------
    max_iter : int, default 100
        Maximum number of iterations.

    w_init : array, shape (n_features,), default None
        Initial value of coefficients.
        If set to ``None``, a zero vector is used instead.

    use_acc : bool, default False
        Extrapolate the iterates based on the past 5 iterates if set to ``True``.
        Can only be used when ``greedy_cd`` is ``False``.

    greedy_cd : bool, default True
        Use a greedy strategy to select features to update in coordinate descent epochs
        if set to ``True``. A cyclic strategy is used otherwise.

    tol : float, default 1e-4
        Tolerance for convergence.

    verbose : bool, default False
        Amount of verbosity. 0/False is silent.
    """

    _datafit_required_attr = ()
    _penalty_required_attr = ("prox_1d", "subdiff_distance")

    def __init__(self, max_iter=100, use_acc=False, greedy_cd=True, tol=1e-4,
                 fit_intercept=True, warm_start=False, verbose=0):
        self.max_iter = max_iter
        self.use_acc = use_acc
        self.greedy_cd = greedy_cd
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.verbose = verbose

    def _solve(self, X, y, datafit, penalty, w_init=None, Xw_init=None):
        # we don't pass Xw_init as the solver uses Gram updates
        # to keep the gradient up-to-date instead of Xw
        n_samples, n_features = X.shape

        if issparse(X):
            scaled_gram = X.T.dot(X)
            scaled_gram = scaled_gram.toarray() / n_samples
            scaled_Xty = X.T.dot(y) / n_samples
        else:
            scaled_gram = X.T @ X / n_samples
            scaled_Xty = X.T @ y / n_samples

        # TODO potential improvement: allow to pass scaled_gram
        # (e.g. for path computation)
        scaled_y_norm2 = np.linalg.norm(y) ** 2 / (2 * n_samples)

        all_features = np.arange(n_features)
        stop_crit = np.inf  # prevent ref before assign
        p_objs_out = []

        w = np.zeros(n_features) if w_init is None else w_init
        grad = - scaled_Xty if w_init is None else scaled_gram @ w_init - scaled_Xty
        opt = penalty.subdiff_distance(w, grad, all_features)

        if self.use_acc:
            if self.greedy_cd:
                warnings.warn(
                    "Anderson acceleration does not work with greedy_cd, " +
                    "set use_acc=False", UserWarning)
            accelerator = AndersonAcceleration(K=5)
            w_acc = np.zeros(n_features)
            grad_acc = np.zeros(n_features)

        for t in range(self.max_iter):
            # check convergences
            stop_crit = np.max(opt)
            if self.verbose:
                p_obj = (0.5 * w @ (scaled_gram @ w) - scaled_Xty @ w +
                         scaled_y_norm2 + penalty.value(w))
                print(
                    f"Iteration {t+1}: {p_obj:.10f}, "
                    f"stopping crit: {stop_crit:.2e}"
                )

            if stop_crit <= self.tol:
                if self.verbose:
                    print(f"Stopping criterion max violation: {stop_crit:.2e}")
                break

            # inplace update of w, grad
            opt = _gram_cd_epoch(scaled_gram, w, grad, penalty, self.greedy_cd)

            # perform Anderson extrapolation
            if self.use_acc:
                w_acc, grad_acc, is_extrapolated = accelerator.extrapolate(w, grad)

                if is_extrapolated:
                    # omit constant term for comparison
                    p_obj_acc = (0.5 * w_acc @ (scaled_gram @ w_acc) -
                                 scaled_Xty @ w_acc + penalty.value(w_acc))
                    p_obj = (0.5 * w @ (scaled_gram @ w) - scaled_Xty @ w
                             + penalty.value(w))
                    if p_obj_acc < p_obj:
                        w[:] = w_acc
                        grad[:] = grad_acc

            # store p_obj
            p_obj = (0.5 * w @ (scaled_gram @ w) - scaled_Xty @ w + scaled_y_norm2 +
                     penalty.value(w))
            p_objs_out.append(p_obj)
        return w, np.array(p_objs_out), stop_crit

    def custom_compatibility_check(self, X, y, datafit, penalty):
        if datafit is not None:
            raise AttributeError(
                "`GramCD` supports only `Quadratic` datafit and fits it implicitly, "
                f"argument `datafit` must be `None`, got {datafit.__class__.__name__}."
            )


@njit
def _gram_cd_epoch(scaled_gram, w, grad, penalty, greedy_cd):
    all_features = np.arange(len(w))
    for cd_iter in all_features:
        # select feature j
        if greedy_cd:
            opt = penalty.subdiff_distance(w, grad, all_features)
            j = np.argmax(opt)
        else:  # cyclic
            j = cd_iter

        # update w_j
        old_w_j = w[j]
        step = 1 / scaled_gram[j, j]  # 1 / lipschitz_j
        w[j] = penalty.prox_1d(old_w_j - step * grad[j], step, j)

        # gradient update with Gram matrix
        if w[j] != old_w_j:
            grad += (w[j] - old_w_j) * scaled_gram[:, j]

    return penalty.subdiff_distance(w, grad, all_features)
