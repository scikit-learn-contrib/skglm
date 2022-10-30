# License: BSD 3 clause

import warnings
import numpy as np
from scipy.sparse import issparse
from scipy.special import expit
from skglm.solvers.prox_newton import ProxNewton

from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, check_consistent_length
from sklearn.linear_model._base import (
    LinearModel, RegressorMixin,
    LinearClassifierMixin, SparseCoefMixin, BaseEstimator
)
from sklearn.utils.extmath import softmax
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier, check_classification_targets

from skglm.utils import compiled_clone
from skglm.solvers import AndersonCD, MultiTaskBCD
from skglm.datafits import Quadratic, Logistic, QuadraticSVC, QuadraticMultiTask
from skglm.penalties import L1, WeightedL1, L1_plus_L2, MCPenalty, IndicatorBox, L2_1


def _glm_fit(X, y, model, datafit, penalty, solver):
    is_classif = isinstance(datafit, (Logistic, QuadraticSVC))
    fit_intercept = solver.fit_intercept

    if is_classif:
        check_classification_targets(y)
        enc = LabelEncoder()
        y = enc.fit_transform(y)
        model.classes_ = enc.classes_
        n_classes_ = len(model.classes_)
        is_sparse = issparse(X)
        if n_classes_ <= 2:
            y = 2 * y - 1
        X = check_array(
            X, accept_sparse="csc", dtype=np.float64, accept_large_sparse=False)
        y = check_array(
            y, ensure_2d=False, dtype=X.dtype.type, accept_large_sparse=False)
        check_consistent_length(X, y)
    else:
        check_X_params = dict(
            dtype=[np.float64, np.float32], order='F',
            accept_sparse='csc', copy=fit_intercept)
        check_y_params = dict(ensure_2d=False, order='F')

        X, y = model._validate_data(
            X, y, validate_separately=(check_X_params, check_y_params))
        X = check_array(X, 'csc', dtype=[np.float64, np.float32],
                        order='F', copy=False, accept_large_sparse=False)
        y = check_array(y, 'csc', dtype=X.dtype.type, order='F', copy=False,
                        ensure_2d=False)

    if y.ndim == 2 and y.shape[1] == 1:
        warnings.warn("DataConversionWarning('A column-vector y"
                      " was passed when a 1d array was expected")
        y = y[:, 0]

    if not hasattr(model, "n_features_in_"):
        model.n_features_in_ = X.shape[1]

    n_samples = X.shape[0]
    if n_samples != y.shape[0]:
        raise ValueError("X and y have inconsistent dimensions (%d != %d)"
                         % (n_samples, y.shape[0]))

    # if not model.warm_start or not hasattr(model, "coef_"):
    if not solver.warm_start or not hasattr(model, "coef_"):
        model.coef_ = None

    if is_classif and n_classes_ > 2:
        model.coef_ = np.empty([len(model.classes_), X.shape[1]])
        if isinstance(datafit, QuadraticSVC):
            model.dual_coef_ = np.empty([len(model.classes_), X.shape[0]])
        model.intercept_ = 0
        multiclass = OneVsRestClassifier(model).fit(X, y)
        model.coef_ = np.array(
            [clf.coef_[0] for clf in multiclass.estimators_])
        if isinstance(datafit, QuadraticSVC):
            model.dual_coef_ = np.array(
                [clf.dual_coef_[0] for clf in multiclass.estimators_])
        model.n_iter_ = max(
            clf.n_iter_ for clf in multiclass.estimators_)
        return model

    if is_classif and n_classes_ <= 2 and isinstance(datafit, QuadraticSVC):
        if is_sparse:
            yXT = (X.T).multiply(y)
            yXT = yXT.tocsc()
        else:
            yXT = (X * y[:, None]).T
        X_ = yXT
    else:
        X_ = X

    n_samples, n_features = X_.shape

    penalty_jit = compiled_clone(penalty)
    datafit_jit = compiled_clone(datafit, to_float32=X.dtype == np.float32)
    if issparse(X):
        datafit_jit.initialize_sparse(X_.data, X_.indptr, X_.indices, y)
    else:
        datafit_jit.initialize(X_, y)

    # if model.warm_start and hasattr(model, 'coef_') and model.coef_ is not None:
    if solver.warm_start and hasattr(model, 'coef_') and model.coef_ is not None:
        if isinstance(datafit, QuadraticSVC):
            w = model.dual_coef_[0, :].copy()
        elif is_classif:
            w = model.coef_[0, :].copy()
        else:
            w = model.coef_.copy()
        if fit_intercept:
            w = np.hstack([w, model.intercept_])
        Xw = X_ @ w[:w.shape[0] - fit_intercept] + fit_intercept * w[-1]
    else:
        # TODO this should be solver.get_init() do delegate the work
        if y.ndim == 1:
            w = np.zeros(n_features + fit_intercept, dtype=X_.dtype)
            Xw = np.zeros(n_samples, dtype=X_.dtype)
        else:  # multitask
            w = np.zeros((n_features + fit_intercept, y.shape[1]), dtype=X_.dtype)
            Xw = np.zeros(y.shape, dtype=X_.dtype)

    # check consistency of weights for WeightedL1
    if isinstance(penalty, WeightedL1):
        if len(penalty.weights) != n_features:
            raise ValueError(
                "The size of the WeightedL1 penalty weights should be n_features, "
                "expected %i, got %i." % (X_.shape[1], len(penalty.weights)))

    coefs, p_obj, kkt = solver.solve(X_, y, datafit_jit, penalty_jit, w, Xw)
    model.coef_, model.stop_crit_ = coefs[:n_features], kkt
    if y.ndim == 1:
        model.intercept_ = coefs[-1] if fit_intercept else 0.
    else:
        model.intercept_ = coefs[-1, :] if fit_intercept else np.zeros(
            y.shape[1])

    model.n_iter_ = len(p_obj)

    if is_classif and n_classes_ <= 2:
        model.coef_ = coefs[np.newaxis, :n_features]
        if isinstance(datafit, QuadraticSVC):
            if is_sparse:
                primal_coef = ((yXT).multiply(model.coef_[0, :])).T
            else:
                primal_coef = (yXT * model.coef_[0, :]).T
            primal_coef = primal_coef.sum(axis=0)
            model.coef_ = np.array(primal_coef).reshape(1, -1)
            model.dual_coef_ = coefs[np.newaxis, :]
    return model


class GeneralizedLinearEstimator(LinearModel):
    r"""Generic generalized linear estimator.

    This estimator takes a penalty and a datafit and runs a coordinate descent solver
    to solve the optimization problem. It handles classification and regression tasks.

    Parameters
    ----------
    datafit : instance of BaseDatafit, optional
        Datafit. If None, `datafit` is initialized as a `Quadratic` datafit.
        `datafit` is replaced by a JIT-compiled instance when calling fit.

    penalty : instance of BasePenalty, optional
        Penalty. If None, `penalty` is initialized as a `L1` penalty.
        `penalty` is replaced by a JIT-compiled instance when calling fit.

    solver : instance of BaseSolver, optional
        Solver. If None, `solver` is initialized as an `AndersonCD` solver.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_features, n_tasks)
        parameter array (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1) or (n_features, n_tasks)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : array, shape (n_tasks,)
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved to reach the specified tolerance.
    """

    def __init__(self, datafit=None, penalty=None, solver=None):
        super(GeneralizedLinearEstimator, self).__init__()
        self.penalty = penalty
        self.datafit = datafit
        self.solver = solver

    def __repr__(self):
        """Get string representation of the estimator.

        Returns
        -------
        repr : str
            String representation.
        """
        return (
            'GeneralizedLinearEstimator(datafit=%s, penalty=%s, alpha=%s)'
            % (self.datafit.__class__.__name__, self.penalty.__class__.__name__,
               self.penalty.alpha))

    def fit(self, X, y):
        """Fit estimator.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        y : array, shape (n_samples,) or (n_samples, n_tasks)
            Target array.

        Returns
        -------
        alphas : array, shape (n_alphas,)
            The alphas along the path where models are computed.

        coefs : array, shape (n_features, n_alphas) or (n_features, n_tasks, n_alphas)
            Coefficients along the path.

        stop_crit : array, shape (n_alphas,)
            Value of stopping criterion at convergence along the path.

        n_iters : array, shape (n_alphas,), optional
            The number of iterations along the path. If return_n_iter is set to `True`.
        """
        self.penalty = self.penalty if self.penalty else L1(1.)
        self.datafit = self.datafit if self.datafit else Quadratic()
        self.solver = self.solver if self.solver else AndersonCD()

        return _glm_fit(X, y, self, self.datafit, self.penalty, self.solver)

    def predict(self, X):
        """Predict target values for samples in X.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data matrix to predict from.

        Returns
        -------
        y_pred : array, shape (n_samples)
            Contain the target values for each sample.
        """
        if isinstance(self.datafit, (Logistic, QuadraticSVC)):
            scores = self._decision_function(X).ravel()
            if len(scores.shape) == 1:
                indices = (scores > 0).astype(int)
            else:
                indices = scores.argmax(axis=1)
            return self.classes_[indices]
        else:
            return self._decision_function(X)

    def get_params(self, deep=False):
        """Get parameters of the estimators including the datafit's and penalty's.

        Parameters
        ----------
        deep : bool
            Whether or not return the parameters for contained subobjects estimators.

        Returns
        -------
        params : dict
            The parameters of the estimator.
        """
        params = super().get_params(deep)
        filtered_types = (float, int, str, np.ndarray)
        penalty_params = [('penalty__', p, getattr(self.penalty, p)) for p in
                          dir(self.penalty) if p[0] != "_" and
                          type(getattr(self.penalty, p)) in filtered_types]
        datafit_params = [('datafit__', p, getattr(self.datafit, p)) for p in
                          dir(self.datafit) if p[0] != "_" and
                          type(getattr(self.datafit, p)) in filtered_types]
        for p_prefix, p_key, p_val in penalty_params + datafit_params:
            params[p_prefix + p_key] = p_val
        return params


class Lasso(LinearModel, RegressorMixin):
    r"""Lasso estimator based on Celer solver and primal extrapolation.

    The optimization objective for Lasso is::

        (1 / (2 * n_samples)) * ||y - X w||^2_2 + alpha * \sum_j |w_j|

    Parameters
    ----------
    alpha : float, optional
        Penalty strength.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions).

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or int
        Amount of verbosity.

    tol : float, optional
        Stopping criterion for the optimization.

    positive : bool, optional
        When set to ``True``, forces the coefficient vector to be positive.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str
        The score used to build the working set. Can be ``fixpoint`` or ``subdiff``.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved to reach the specified tolerance.

    See Also
    --------
    WeightedLasso : Weighted Lasso regularization.
    MCPRegression : Sparser regularization than L1 norm.
    """

    def __init__(self, alpha=1., max_iter=50, max_epochs=50_000, p0=10, verbose=0,
                 tol=1e-4, positive=False, fit_intercept=True, warm_start=False,
                 ws_strategy="subdiff"):
        super().__init__()
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.ws_strategy = ws_strategy
        self.positive = positive
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.verbose = verbose

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
        # TODO: Add Gram solver
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return _glm_fit(X, y, self, Quadratic(), L1(self.alpha, self.positive), solver)

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **params):
        """Compute Lasso path.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        y : array, shape (n_samples,)
            Target vector.

        alphas : array, shape (n_alphas,)
            Grid of alpha.

        coef_init : array, shape (n_features,), optional
            If warm_start is enabled, the optimization problem restarts from coef_init.

        return_n_iter : bool
            Returns the number of iterations along the path.

        **params : kwargs
            All parameters supported by path.

        Returns
        -------
        alphas : array, shape (n_alphas,)
            The alphas along the path where models are computed.

        coefs : array, shape (n_features, n_alphas)
            Coefficients along the path.

        stop_crit : array, shape (n_alphas,)
            Value of stopping criterion at convergence along the path.

        n_iters : array, shape (n_alphas,), optional
            The number of iterations along the path. If return_n_iter is set to `True`.
        """
        penalty = compiled_clone(L1(self.alpha, self.positive))
        datafit = compiled_clone(Quadratic(), to_float32=X.dtype == np.float32)
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return solver.path(X, y, datafit, penalty, alphas, coef_init, return_n_iter)


class WeightedLasso(LinearModel, RegressorMixin):
    r"""WeightedLasso estimator based on Celer solver and primal extrapolation.

    The optimization objective for WeightedLasso is::

        (1 / (2 * n_samples)) * ||y - X w||^2_2 + alpha * \sum_j weights_j |w_j|

    Parameters
    ----------
    alpha : float, optional
        Penalty strength.

    weights : array, shape (n_features,), optional (default=None)
        Positive weights used in the L1 penalty part of the Lasso
        objective. If None, weights equal to 1 are used.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions).

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or int
        Amount of verbosity.

    tol : float, optional
        Stopping criterion for the optimization.

    positive : bool, optional
        When set to ``True``, forces the coefficient vector to be positive.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str
        The score used to build the working set. Can be ``fixpoint`` or ``subdiff``.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved to reach the specified tolerance.

    See Also
    --------
    MCPRegression : Sparser regularization than L1 norm.
    Lasso : Unweighted Lasso regularization.

    Notes
    -----
    Supports weights equal to 0, i.e. unpenalized features.
    """

    def __init__(self, alpha=1., weights=None, max_iter=50, max_epochs=50_000, p0=10,
                 verbose=0, tol=1e-4, positive=False, fit_intercept=True,
                 warm_start=False, ws_strategy="subdiff"):
        super().__init__()
        self.alpha = alpha
        self.weights = weights
        self.tol = tol
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.ws_strategy = ws_strategy
        self.positive = positive
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.verbose = verbose

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **params):
        """Compute Weighted Lasso path.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        y : array, shape (n_samples,)
            Target vector.

        alphas : array, shape (n_alphas,)
            Grid of alpha.

        coef_init : array, shape (n_features,), optional
            If warm_start is enabled, the optimization problem restarts from coef_init.

        return_n_iter : bool
            Returns the number of iterations along the path.

        **params : kwargs
            All parameters supported by path.

        Returns
        -------
        alphas : array, shape (n_alphas,)
            The alphas along the path where models are computed.

        coefs : array, shape (n_features, n_alphas)
            Coefficients along the path.

        stop_crit : array, shape (n_alphas,)
            Value of stopping criterion at convergence along the path.

        n_iters : array, shape (n_alphas,), optional
            The number of iterations along the path. If return_n_iter is set to `True`.
        """
        weights = np.ones(X.shape[1]) if self.weights is None else self.weights
        if X.shape[1] != len(weights):
            raise ValueError("The number of weights must match the number of \
                              features. Got %s, expected %s." % (
                len(weights), X.shape[1]))
        penalty = compiled_clone(WeightedL1(self.alpha, weights, self.positive))
        datafit = compiled_clone(Quadratic(), to_float32=X.dtype == np.float32)
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return solver.path(X, y, datafit, penalty, alphas, coef_init, return_n_iter)

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
        if self.weights is None:
            warnings.warn('Weights are not provided, fitting with Lasso penalty')
            penalty = L1(self.alpha, self.positive)
        else:
            penalty = WeightedL1(self.alpha, self.weights, self.positive)
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return _glm_fit(X, y, self, Quadratic(), penalty, solver)


class ElasticNet(LinearModel, RegressorMixin):
    r"""Elastic net estimator.

    The optimization objective for Elastic net is::

        (1 / (2 * n_samples)) * ||y - X w||^2_2 + l1_ratio * alpha * sum_j |w_j| \
        + (1 - l1_ratio) * alpha / 2 sum_j w_j ** 2

    Parameters
    ----------
    alpha : float, optional
        Penalty strength.

    l1_ratio : float, default=0.5
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions).

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or int
        Amount of verbosity.

    tol : float, optional
        Stopping criterion for the optimization.

    positive : bool, optional
        When set to ``True``, forces the coefficient vector to be positive.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str
        The score used to build the working set. Can be ``fixpoint`` or ``subdiff``.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved to reach the specified tolerance.

    See Also
    --------
    Lasso : Lasso regularization.
    """

    def __init__(self, alpha=1., l1_ratio=0.5, max_iter=50, max_epochs=50_000, p0=10,
                 verbose=0, tol=1e-4, positive=False, fit_intercept=True,
                 warm_start=False, ws_strategy="subdiff"):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.ws_strategy = ws_strategy
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.warm_start = warm_start
        self.verbose = verbose

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **params):
        """Compute Elastic Net path.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        y : array, shape (n_samples,)
            Target vector.

        alphas : array, shape (n_alphas,)
            Grid of alpha.

        coef_init : array, shape (n_features,), optional
            If warm_start is enabled, the optimization problem restarts from coef_init.

        return_n_iter : bool
            Returns the number of iterations along the path.

        **params : kwargs
            All parameters supported by path.

        Returns
        -------
        alphas : array, shape (n_alphas,)
            The alphas along the path where models are computed.

        coefs : array, shape (n_features, n_alphas)
            Coefficients along the path.

        stop_crit : array, shape (n_alphas,)
            Value of stopping criterion at convergence along the path.

        n_iters : array, shape (n_alphas,), optional
            The number of iterations along the path. If return_n_iter is set to `True`.
        """
        penalty = compiled_clone(L1_plus_L2(self.alpha, self.l1_ratio, self.positive))
        datafit = compiled_clone(Quadratic(), to_float32=X.dtype == np.float32)
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return solver.path(X, y, datafit, penalty, alphas, coef_init, return_n_iter)

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
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return _glm_fit(X, y, self, Quadratic(),
                        L1_plus_L2(self.alpha, self.l1_ratio, self.positive), solver)


class MCPRegression(LinearModel, RegressorMixin):
    r"""Linear regression with MCP penalty estimator.

    The optimization objective for MCPRegression is, with x >= 0::

        pen(x) = alpha * x - x^2 / (2 * gamma) if x =< gamma * alpha
                 gamma * alpha ** 2 / 2        if x > gamma * alpha

        obj = (1 / (2 * n_samples)) * ||y - X w||^2_2 + pen(|w_j|)

    For more details see
    Coordinate descent algorithms for nonconvex penalized regression,
    with applications to biological feature selection, Breheny and Huang.

    Parameters
    ----------
    alpha : float, optional
        Penalty strength.

    gamma : float, default=3
        If gamma = 1, the prox of MCP is a hard thresholding.
        If gamma = np.inf it is a soft thresholding.
        Should be larger than (or equal to) 1.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions).

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or int
        Amount of verbosity.

    tol : float, optional
        Stopping criterion for the optimization.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str
        The score used to build the working set. Can be ``fixpoint`` or ``subdiff``.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved to reach the specified tolerance.

    See Also
    --------
    Lasso : Lasso regularization.
    """

    def __init__(self, alpha=1., gamma=3, max_iter=50, max_epochs=50_000, p0=10,
                 verbose=0, tol=1e-4, fit_intercept=True, warm_start=False,
                 ws_strategy="subdiff"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.ws_strategy = ws_strategy
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.verbose = verbose

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **params):
        """Compute MCPRegression path.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        y : array, shape (n_samples,)
            Target vector.

        alphas : array, shape (n_alphas,)
            Grid of alpha.

        coef_init : array, shape (n_features,), optional
            If warm_start is enabled, the optimization problem restarts from coef_init.

        return_n_iter : bool
            Returns the number of iterations along the path.

        **params : kwargs
            All parameters supported by path.

        Returns
        -------
        alphas : array, shape (n_alphas,)
            The alphas along the path where models are computed.

        coefs : array, shape (n_features, n_alphas)
            Coefficients along the path.

        stop_crit : array, shape (n_alphas,)
            Value of stopping criterion at convergence along the path.

        n_iters : array, shape (n_alphas,), optional
            The number of iterations along the path. If return_n_iter is set to `True`.
        """
        penalty = compiled_clone(MCPenalty(self.alpha, self.gamma))
        datafit = compiled_clone(Quadratic(), to_float32=X.dtype == np.float32)
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return solver.path(X, y, datafit, penalty, alphas, coef_init, return_n_iter)

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
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return _glm_fit(X, y, self, Quadratic(), MCPenalty(self.alpha, self.gamma),
                        solver)


class SparseLogisticRegression(LinearClassifierMixin, SparseCoefMixin, BaseEstimator):
    r"""Sparse Logistic regression estimator.

    The optimization objective for sparse Logistic regression is::

        mean(log(1 + exp(-y_i x_i^T w))) + alpha * ||w||_1

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength; must be a positive float.

    tol : float, optional
        Stopping criterion for the optimization.

    max_iter : int, optional
        The maximum number of outer iterations (subproblem definitions).

    max_epochs : int
        Maximum number of prox Newton iterations on each subproblem.

    verbose : bool or int
        Amount of verbosity.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes, )
        A list of class labels known to the classifier.

    coef_ : ndarray, shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.

        `coef_` is of shape (1, n_features) when the given problem is binary.

    intercept_ :  ndarray, shape (1,) or (n_classes,)
        constant term in decision function. Not handled yet.

    n_iter_ : int
        Number of subproblems solved to reach the specified tolerance.
    """

    def __init__(self, alpha=1.0, tol=1e-4, max_iter=20, max_epochs=1_000, verbose=0,
                 fit_intercept=True, warm_start=False):
        super().__init__()
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start

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
        solver = ProxNewton(
            max_iter=self.max_iter, max_pn_iter=self.max_epochs, tol=self.tol,
            fit_intercept=self.fit_intercept, warm_start=self.warm_start,
            verbose=self.verbose)
        return _glm_fit(X, y, self, Logistic(), L1(self.alpha), solver)

    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.
        For a multi_class problem, a one-vs-rest approach, i.e calculate the probability
        of each class assuming it to be positive using the logistic function.
        and normalize these values across all the classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        check_is_fitted(self)
        if len(self.classes_) > 2:
            # Code taken from https://github.com/scikit-learn/scikit-learn/
            # blob/c900ad385cecf0063ddd2d78883b0ea0c99cd835/sklearn/
            # linear_model/_base.py#L458
            def _predict_proba_lr(X):
                """Probability estimation for OvR logistic regression.

                Positive class probabilities are computed as
                1. / (1. + np.exp(-self.decision_function(X)));
                multiclass is handled by normalizing that over all classes.
                """
                prob = self.decision_function(X)
                expit(prob, out=prob)
                if prob.ndim == 1:
                    return np.vstack([1 - prob, prob]).T
                else:
                    # OvR normalization, like LibLinear's predict_probability
                    prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
                    return prob
            # OvR normalization, like LibLinear's
            return _predict_proba_lr(X)
        else:
            decision = self.decision_function(X)
            if decision.ndim == 1:
                # Workaround for multi_class="multinomial" and binary outcomes
                # which requires softmax prediction with only a 1D decision.
                decision_2d = np.c_[-decision, decision]
            else:
                decision_2d = decision
            return softmax(decision_2d, copy=False)


class LinearSVC(LinearClassifierMixin, SparseCoefMixin, BaseEstimator):
    r"""LinearSVC estimator, with hinge loss.

    The optimization objective for LinearSVC is::

        C * \sum_i max(0, 1 - y_i beta.T X[i, :]) + 1 / 2 * ||beta||^2

    i.e. hinge datafit loss (non-smooth) + l2 regularization (smooth)

    To solve this, we solve the dual optimization problem to stay in our
    framework of smooth datafit and non-smooth penalty.
    The dual optimization problem of SVC is::

        1 / 2 * ||(y X).T w||^2_2 - \sum_i w_i + \sum_i ind(0 <= w_i <= C)

    The primal-dual relation is given by::
        w = \sum_i y_i * w_i * X[i, :]

    Parameters
    ----------
    C : float, optional
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions).

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or int
        Amount of verbosity.

    tol : float, optional
        Stopping criterion for the optimization.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str
        The score used to build the working set. Can be ``fixpoint`` or ``subdiff``.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        constant term in decision function.

    dual_ : array, shape (n_samples,)
        dual of the solution.

    n_iter_ : int
        Number of subproblems solved to reach the specified tolerance.
    """

    def __init__(self, C=1., max_iter=50, max_epochs=50_000, p0=10,
                 verbose=0, tol=1e-4, fit_intercept=True, warm_start=False,
                 ws_strategy="subdiff"):
        super().__init__()
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.ws_strategy = ws_strategy
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.verbose = verbose

    def fit(self, X, y):
        """Fit LinearSVC classifier.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        y : array, shape (n_samples,)
            Target vector.

        Returns
        -------
        self
            Fitted estimator.
        """
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=False,
            warm_start=self.warm_start, verbose=self.verbose)
        return _glm_fit(X, y, self, QuadraticSVC(), IndicatorBox(self.C), solver)

    # TODO add predict_proba for LinearSVC


class MultiTaskLasso(LinearModel, RegressorMixin):
    r"""MultiTaskLasso estimator.

    The optimization objective for MultiTaskLasso is::

        (1 / (2 * n_samples)) * ||y - X W||^2_2 + alpha * ||W||_{21}

    Parameters
    ----------
    alpha : float, optional
        Regularization strength (constant that multiplies the L21 penalty).

    copy_X : bool, optional (default=True)
        If True, X will be copied; else, it may be overwritten.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions).

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or int
        Amount of verbosity.

    tol : float, optional
        Stopping criterion for the optimization.

    positive : bool, optional
        When set to ``True``, forces the coefficient vector to be positive.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str
        The score used to build the working set. Can be ``fixpoint`` or ``subdiff``.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved by Celer to reach the specified tolerance.
    """

    def __init__(self, alpha=1., copy_X=True, max_iter=50, max_epochs=50_000, p0=10,
                 verbose=0, tol=1e-4, positive=False, fit_intercept=True, 
                 warm_start=False, ws_strategy="subdiff"):
        self.tol = tol
        self.alpha = alpha
        self.copy_X = copy_X
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.positive = positive
        self.p0 = p0
        self.ws_strategy = ws_strategy
        self.max_epochs = max_epochs
        self.verbose = verbose

    def fit(self, X, Y):
        """Fit MultiTaskLasso model.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Design matrix.

        Y : ndarray, shape (n_samples, n_tasks)
            Observation matrix.

        Returns
        -------
        self :
            The fitted estimator.
        """
        # Below is copied from sklearn, with path replaced by our path.
        # Need to validate separately here.
        # We can't pass multi_output=True because that would allow y to be csr.
        check_X_params = dict(dtype=[np.float64, np.float32], order='F',
                              accept_sparse='csc',
                              copy=self.copy_X and self.fit_intercept)
        check_Y_params = dict(ensure_2d=False, order='F')
        X, Y = self._validate_data(X, Y, validate_separately=(check_X_params,
                                                              check_Y_params))
        Y = Y.astype(X.dtype)

        if Y.ndim == 1:
            raise ValueError("For mono-task outputs, use Lasso")

        n_samples = X.shape[0]

        if n_samples != Y.shape[0]:
            raise ValueError("X and Y have inconsistent dimensions (%d != %d)"
                             % (n_samples, Y.shape[0]))

        # X, Y, X_offset, Y_offset, X_scale = _preprocess_data(
        #     X, Y, self.fit_intercept, copy=False)

        # TODO handle and test warm start for MTL
        if not self.warm_start or not hasattr(self, "coef_"):
            self.coef_ = None

        datafit_jit = compiled_clone(QuadraticMultiTask(), X.dtype == np.float32)
        penalty_jit = compiled_clone(L2_1(self.alpha, self.positive), X.dtype == np.float32)

        solver = MultiTaskBCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        W, obj_out, kkt = solver.solve(X, Y, datafit_jit, penalty_jit)

        self.coef_ = W[:X.shape[1], :].T
        self.intercept_ = self.fit_intercept * W[-1, :]
        self.stopping_crit = kkt
        self.n_iter_ = len(obj_out)

        return self

    def path(self, X, Y, alphas, coef_init=None, return_n_iter=False, **params):
        """Compute MultitaskLasso path.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        Y : array, shape (n_samples, n_tasks)
            Target matrix.

        alphas : array, shape (n_alphas,)
            Grid of alpha.

        coef_init : array, shape (n_features,), optional
            If warm_start is enabled, the optimization problem restarts from coef_init.

        return_n_iter : bool
            Returns the number of iterations along the path.

        **params : kwargs
            All parameters supported by path.

        Returns
        -------
        alphas : array, shape (n_alphas,)
            The alphas along the path where models are computed.

        coefs : array, shape (n_features, n_tasks, n_alphas)
            Coefficients along the path.

        stop_crit : array, shape (n_alphas,)
            Value of stopping criterion at convergence along the path.

        n_iters : array, shape (n_alphas,), optional
            The number of iterations along the path. If return_n_iter is set to `True`.
        """
        datafit = compiled_clone(QuadraticMultiTask(), to_float32=X.dtype == np.float32)
        penalty = compiled_clone(L2_1(self.alpha, self.positive))
        solver = MultiTaskBCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return solver.path(X, Y, datafit, penalty, alphas, coef_init, return_n_iter)
