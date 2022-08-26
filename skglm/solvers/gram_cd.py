import warnings
import numpy as np
from numba import njit
from scipy.sparse import issparse

from skglm.utils import AndersonAcceleration


def gram_cd_solver(X, y, penalty, max_iter=100, w_init=None,
                   use_acc=True, greedy_cd=True, tol=1e-4, verbose=False):
    r"""Run coordinate descent while keeping the gradients up-to-date with Gram updates.

    This solver should be used when n_features < n_samples, and computes the
    (n_features, n_features) Gram matrix which comes with an overhead. It is  only
    suited to Quadratic datafits.

    It minimizes::
        1 / (2*n_samples) * norm(y - Xw)**2 + penalty(w)

    which can be rewritten as::
        w.T @ Q @ w / (2*n_samples) - q.T @ w / n_samples + penalty(w)

    where::
        Q = X.T @ X (gram matrix), and q = X.T @ y

    Parameters
    ----------
    X : array or sparse CSC matrix, shape (n_samples, n_features)
        Design matrix.

    y : array, shape (n_samples,)
        Target vector.

    penalty : instance of BasePenalty
        Penalty object.

    max_iter : int, default 100
        Maximum number of iterations.

    w_init : array, shape (n_features,), default None
        Initial value of coefficients.
        If set to None, a zero vector is used instead.

    use_acc : bool, default True
        Extrapolate the iterates based on the past 5 iterates if set to True.

    greedy_cd : bool, default True
        Use a greedy strategy to select features to update in coordinate descent epochs
        if set to True. A cyclic strategy is used otherwise.

    tol : float, default 1e-4
        Tolerance for convergence.

    verbose : bool, default False
        Amount of verbosity. 0/False is silent.

    Returns
    -------
    w : array, shape (n_features,)
        Solution that minimizes the problem defined by datafit and penalty.

    objs_out : array, shape (n_iter,)
        The objective values at every outer iteration.

    stop_crit : float
        The value of the stopping criterion when the solver stops.
    """
    n_samples, n_features = X.shape

    if issparse(X):
        scaled_gram = X.T.dot(X)
        scaled_gram = scaled_gram.toarray() / n_samples
        scaled_Xty = X.T.dot(y) / n_samples
    else:
        scaled_gram = X.T @ X / n_samples
        scaled_Xty = X.T @ y / n_samples
    # TODO potential improvement: allow to pass scaled_gram (e.g. for path computation)

    scaled_y_norm2 = np.linalg.norm(y)**2 / (2*n_samples)

    all_features = np.arange(n_features)
    stop_crit = np.inf  # prevent ref before assign
    p_objs_out = []

    w = np.zeros(n_features) if w_init is None else w_init
    grad = - scaled_Xty if w_init is None else scaled_gram @ w_init - scaled_Xty
    opt = penalty.subdiff_distance(w, grad, all_features)

    if use_acc:
        if greedy_cd:
            warnings.warn(
                "Anderson acceleration does not work with greedy_cd, set use_acc=False",
                UserWarning)
        accelerator = AndersonAcceleration(K=5)
        w_acc = np.zeros(n_features)
        grad_acc = np.zeros(n_features)

    for t in range(max_iter):
        # check convergences
        stop_crit = np.max(opt)
        if verbose:
            p_obj = (0.5 * w @ (scaled_gram @ w) - scaled_Xty @ w +
                     scaled_y_norm2 + penalty.value(w))
            print(
                f"Iteration {t+1}: {p_obj:.10f}, "
                f"stopping crit: {stop_crit:.2e}"
            )

        if stop_crit <= tol:
            if verbose:
                print(f"Stopping criterion max violation: {stop_crit:.2e}")
            break

        # inplace update of w, XtXw
        opt = _gram_cd_epoch(scaled_gram, w, grad, penalty, greedy_cd)

        # perform Anderson extrapolation
        if use_acc:
            w_acc, grad_acc, is_extrapolated = accelerator.extrapolate(w, grad)

            if is_extrapolated:
                # omit constant term for comparison
                p_obj_acc = (0.5 * w_acc @ (scaled_gram @ w_acc) - scaled_Xty @ w_acc +
                             penalty.value(w_acc))
                p_obj = 0.5 * w @ (scaled_gram @ w) - scaled_Xty @ w + penalty.value(w)
                if p_obj_acc < p_obj:
                    w[:] = w_acc
                    grad[:] = grad_acc

        # store p_obj
        p_obj = (0.5 * w @ (scaled_gram @ w) - scaled_Xty @ w + scaled_y_norm2 +
                 penalty.value(w))
        p_objs_out.append(p_obj)
    return w, np.array(p_objs_out), stop_crit


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
