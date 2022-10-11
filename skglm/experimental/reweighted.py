import numpy as np
from numpy.linalg import norm
from skglm.estimators import Lasso


def _reweight_by_coefficient_norm(coef):
    nrm = np.sqrt(norm(coef))
    return 1 / (2 * nrm + np.finfo(float).eps)


class ReweightedLasso(Lasso):
    r"""Reweighted Lasso estimator.

    The objective reads::
        ||y-Xw||_2^2 / (2 * n_samples) + \lambda \sqrt{||w||}

    This estimator iteratively solves a non-convex objective by iteratively solving
    convex surrogates. They are obtained by iteratively re-weighting the \ell_1-norm
    by a penalty applied to the coefficients.

    Parameters
    ----------
    alpha : float, optional
        Penalty strength.

    n_reweights : int, optional
        Number of reweighting iterations.

    args: list
        Parameters for the inner Lasso estimator.
        Note that `warm_start` is consistently set to `True` for performance reasons.

    kwargs: dict
        Keyword parameters for the inner Lasso estimator.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    loss_history_ : list
        Objective history after every reweighting iteration

    References
    ----------
    .. [1] Candès et al. (2007), Enhancing sparsity by reweighted l1 minimization
           https://web.stanford.edu/~boyd/papers/pdf/rwl1.pdf
    """

    def __init__(self, alpha=1., n_reweights=5, *args, **kwargs):
        # Warm start is crucial to ensure the solution of surrogate problem i
        # is used as the starting coefficient to solve surrogate i+1, hence enabling
        # speed gains
        super().__init__(alpha=alpha, warm_start=True, *args, **kwargs)
        self.n_reweights = n_reweights

    def fit(self, X, y, reweight_penalty=_reweight_by_coefficient_norm):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        reweight_penalty: Callable, optional
            Callable to compute the weights from the coefficients.
            By default, it is the \ell_0.5 norm of the coefficients

        Returns
        -------
        self :
            Fitted estimator.
        """
        self.loss_history_ = []
        n_samples, n_features = X.shape
        weights = np.ones(n_features)
        # XXX: dot product X @ w is slow in high-dimension, to be improved
        objective = (lambda w: np.sum((y - X @ w) ** 2) / (2 * n_samples)
                     + self.alpha * np.sqrt(norm(w)))

        for l in range(self.n_reweights):
            # trick: rescaling the weights (XXX: sparse X would become dense?)
            scaled_X = X / weights
            super().fit(scaled_X, y)
            scaled_coef = self.coef_ / weights

            # updating the weights
            weights = reweight_penalty(scaled_coef)

            loss = objective(scaled_coef)
            self.loss_history_.append(loss)

            if self.verbose:
                print("#" * 10)
                print(f"[REWEIGHT] iteration {l} :: loss {loss}")
                print("#" * 10)

        self.coef_ = scaled_coef

        return self



