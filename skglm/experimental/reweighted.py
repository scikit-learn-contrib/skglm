import numpy as np
from skglm.estimators import GeneralizedLinearEstimator
from skglm.penalties import WeightedL1


def _L05_objective(coef):
    return np.sqrt(np.abs(coef))


def _L05_derivative(coef):
    return 1. / (2. * np.sqrt(np.abs(coef)) + np.finfo(float).eps)


def _log_sum_objective(coef):
    return np.log(np.abs(coef))


def _log_sum_derivative(coef):
    return 1. / (np.abs(coef))


class IterativeReweightedL1(GeneralizedLinearEstimator):
    r"""Reweighted L1-norm estimator.

    This estimator iteratively solves a non-convex objective by iteratively solving
    convex surrogates. They are obtained by iteratively re-weighting the \ell_1-norm
    by a penalty applied to the coefficients.

    Parameters
    ----------
    alpha : float, optional
        Penalty strength.

    datafit : instance of BaseDatafit, optional
        Datafit. If None, `datafit` is initialized as a `Quadratic` datafit.
        `datafit` is replaced by a JIT-compiled instance when calling fit.

    solver : instance of BaseSolver, optional
        Solver. If None, `solver` is initialized as an `AndersonCD` solver.

    n_reweights : int, optional
        Number of reweighting iterations.

    pen_obj : Callable, optional
        Compute the concave objective.

    pen_weight: Callable, optional
        Compute the penalty weights from the coefficients.

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

    def __init__(self, alpha, datafit=None, solver=None, n_reweights=5,
                 pen_obj=_L05_objective, pen_weight=_L05_derivative):
        super().__init__(
            datafit=datafit, penalty=WeightedL1(alpha, np.ones(1)), solver=solver)
        self.n_reweights = n_reweights
        self.pen_obj = pen_obj
        self.pen_weight = pen_weight

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self :
            Fitted estimator.
        """
        self.loss_history_ = []
        n_features = X.shape[1]
        self.penalty.weights = np.ones(n_features)

        for iter_reweight in range(self.n_reweights):
            super().fit(X, y)
            self.penalty.weights = self.pen_weight(self.coef_)

            # TODO: dot product X @ w is slow in high-dimension, to be improved
            loss = (self.datafit.value(y, self.coef_, X @ self.coef_)
                    + self.penalty.alpha * np.sum(self.pen_obj(self.coef_)))
            self.loss_history_.append(loss)

            if self.solver.verbose:
                print("#" * 10)
                print(f"[REWEIGHT] iteration {iter_reweight} :: loss {loss}")
                print("#" * 10)

        return self
