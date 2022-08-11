# License: BSD 3 clause

import numpy as np
from scipy.sparse import issparse
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.multiclass import check_classification_targets
from sklearn.linear_model import MultiTaskLasso as MultiTaskLasso_sklearn
from sklearn.linear_model._base import LinearModel
from sklearn.svm import LinearSVC as LinearSVC_sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model._base import _preprocess_data

from skglm.penalties import (
    L1, WeightedL1, L1_plus_L2, MCPenalty, IndicatorBox, L2_1
)
from skglm.datafits import (
    Quadratic, Logistic, QuadraticSVC, QuadraticMultiTask
)
from skglm.utils import compiled_clone
from skglm.solvers import cd_solver_path, bcd_solver_path


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

    is_classif : bool, optional
        Whether the task is classification or regression. Used for input target
        validation.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions).

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    tol : float, optional
        Stopping criterion for the optimization.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str
        The score used to build the working set. Can be ``fixpoint`` or ``subdiff``.

    verbose : bool or int
        Amount of verbosity.

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

    def __init__(self, datafit=None, penalty=None, is_classif=False, max_iter=100,
                 max_epochs=50_000, p0=10, tol=1e-4, fit_intercept=False,
                 warm_start=False, ws_strategy="subdiff", verbose=0):
        super(GeneralizedLinearEstimator, self).__init__()
        self.is_classif = is_classif
        self.tol = tol
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.p0 = p0
        self.ws_strategy = ws_strategy
        self.penalty = penalty
        self.datafit = datafit

    def __repr__(self):
        """Get string representation of the estimator.

        Returns
        -------
        repr : str
            String representation.
        """
        return (
            'GeneralizedLinearEstimator(datafit=%s, penalty=%s, alpha=%s, classif=%s)'
            % (self.datafit.__class__.__name__, self.penalty.__class__.__name__,
               self.penalty.alpha, self.is_classif))

    def path(self, X, y, alphas, coef_init=None, return_n_iter=False, **params):
        """Compute regularization path.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        y : array, shape (n_samples,) or (n_samples, n_tasks)
            Target array.

        alphas : array, shape (n_alphas,)
            Grid of alpha.

        coef_init : array, shape (n_features,) or (n_features, n_tasks), optional
            If warm_start is enabled, the optimization problem restarts from coef_init.

        return_n_iter : bool
            Returns the number of iterations along the path.

        **params : kwargs
            All parameters supported by path.

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
        penalty = compiled_clone(self.penalty)
        datafit = compiled_clone(self.datafit, to_float32=X.dtype == np.float32)

        path_func = cd_solver_path if y.ndim == 1 else bcd_solver_path
        return path_func(
            X, y, datafit, penalty, alphas=alphas,
            coef_init=coef_init, max_iter=self.max_iter,
            return_n_iter=return_n_iter, max_epochs=self.max_epochs, p0=self.p0,
            tol=self.tol, use_acc=True, ws_strategy=self.ws_strategy,
            fit_intercept=self.fit_intercept, verbose=self.verbose)

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
        if y is None:
            raise ValueError("requires y to be passed, but the target y is None")

        self.penalty = self.penalty if self.penalty else L1(1.)
        self.datafit = self.datafit if self.datafit else Quadratic()

        if isinstance(self.penalty, WeightedL1):
            if len(self.penalty.weights) != X.shape[1]:
                raise ValueError(
                    "The size of the WeightedL1 penalty should be n_features")

        if isinstance(self.datafit, Logistic):
            X = check_array(
                X,
                accept_sparse="csr",
                dtype=np.float64)

            y = check_array(y, ensure_2d=False, dtype=None)
            check_consistent_length(X, y)
        else:
            check_X_params = dict(
                dtype=[np.float64, np.float32], order='F',
                accept_sparse='csc', copy=self.fit_intercept)
            check_y_params = dict(ensure_2d=False, order='F')

            X, y = self._validate_data(
                X, y, validate_separately=(check_X_params, check_y_params))

        self.classes_ = None
        n_classes_ = 0

        if self.is_classif:
            check_classification_targets(y)
            enc = LabelEncoder()
            y = enc.fit_transform(y)
            self.classes_ = enc.classes_
            n_classes_ = len(self.classes_)

        if not hasattr(self, "n_features_in_"):
            self.n_features_in_ = X.shape[1]

        is_sparse = issparse(X)
        if isinstance(self.datafit, (QuadraticSVC, Logistic)) and n_classes_ <= 2:
            y = 2 * y - 1
            if is_sparse and isinstance(self.datafit, Logistic):
                yXT = (X.T).multiply(y)
            else:
                yXT = (X * y[:, None]).T

        n_samples, n_features = X.shape
        if n_samples != y.shape[0]:
            raise ValueError("X and y have inconsistent dimensions (%d != %d)"
                             % (n_samples, y.shape[0]))

        # X, y, X_offset, y_offset, X_scale = _preprocess_data(
        #     X, y, self.fit_intercept, copy=False)

        if not self.warm_start or not hasattr(self, "coef_"):
            self.coef_ = None

        X_ = yXT if isinstance(self.datafit, QuadraticSVC) else X

        _, coefs, kkt = self.path(
            X_, y, alphas=[self.penalty.alpha],
            coef_init=self.coef_, max_iter=self.max_iter,
            max_epochs=self.max_epochs, p0=self.p0, verbose=self.verbose,
            tol=self.tol, ws_strategy=self.ws_strategy)

        self.coef_, self.stop_crit_ = coefs[:n_features, 0], kkt[-1]
        self.n_iter_ = len(kkt)

        if self.fit_intercept:
            self.intercept_ = coefs[-1, 0]
        else:
            self.intercept_ = 0.

        if isinstance(self.datafit, QuadraticSVC):
            if n_classes_ <= 2:
                self.coef_ = coefs.T
                if is_sparse:
                    primal_coef = ((yXT).multiply(self.coef_[0, :])).T
                else:
                    primal_coef = (yXT * self.coef_[0, :]).T
                primal_coef = primal_coef.sum(axis=0)
                self.coef_ = np.array(primal_coef).reshape(1, -1)
            elif n_classes_ > 2:
                self.coef_ = np.empty([len(self.classes_), X.shape[1]])
                self.intercept_ = 0
                multiclass = OneVsRestClassifier(self).fit(X, y)
                self.coef_ = np.array([clf.coef_[0]
                                       for clf in multiclass.estimators_])
                self.n_iter_ = max(
                    clf.n_iter_ for clf in multiclass.estimators_)
        elif isinstance(self.datafit, Logistic):
            self.coef_ = coefs.T
        return self

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
        if self.is_classif:
            scores = self._decision_function(X).ravel()
            if len(scores.shape) == 1:
                indices = (scores > 0).astype(int)
            else:
                indices = scores.argmax(axis=1)
            return self.classes_[indices]
        else:
            return self._decision_function(X)

    def get_params(self, deep=True):
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
        filtered_types = (np.float_, np.int_, np.string_, np.ndarray)
        if deep:
            penalty_params = self.penalty.get_params()
            for k, v in penalty_params.items():
                params['penalty__' + k] = v

            datafit_params = [('datafit__', p, getattr(self.datafit, p)) for p in
                              dir(self.datafit) if p[0] != "_" and
                              type(getattr(self.datafit, p)) in filtered_types]
            for p_prefix, p_key, p_val in datafit_params:
                params[p_prefix + p_key] = p_val
        return params

    def set_params(self, **params):
        """Set parameters of the estimators.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        super().set_params(**params)
        self.penalty.set_params(**params)
        return self

    def score(self, X, y):
        """Return the score of the estimator.

        This corresponds to themean accuracy on the given test data and labels
        if the estimator performs classification and return the coefficient
        of determination if the estimation performs regression.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        Returns
        -------
        score : float
            Mean accuracy or :math:`R^2` ``self.predict(X)`` wrt. `y`.
        """
        y_pred = self.predict(X)
        if self.is_classif:
            return accuracy_score(y, y_pred)
        else:
            return r2_score(y, y_pred)


class Lasso(GeneralizedLinearEstimator):
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

    def __init__(self, alpha=1., max_iter=100, max_epochs=50_000, p0=10,
                 verbose=0, tol=1e-4, fit_intercept=False,
                 warm_start=False, ws_strategy="subdiff"):
        super().__init__(
            datafit=Quadratic(), penalty=L1(alpha=alpha), is_classif=False,
            max_iter=max_iter, max_epochs=max_epochs, p0=p0, tol=tol,
            fit_intercept=fit_intercept,
            warm_start=warm_start, ws_strategy=ws_strategy, verbose=verbose)
        self.alpha = alpha

    def get_params(self, deep=False):
        """Get parameters of the estimators.

        Parameters
        ----------
        deep : bool
            Whether or not return the parameters for contained subobjects estimators.

        Returns
        -------
        params : dict
            The parameters of the estimator.
        """
        return super().get_params(deep)


class WeightedLasso(GeneralizedLinearEstimator):
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

    def __init__(self, alpha=1., weights=None, max_iter=100, max_epochs=50_000, p0=10,
                 verbose=0, tol=1e-4, fit_intercept=False,
                 warm_start=False, ws_strategy="subdiff"):
        if weights is None:
            # Warning weights should not be None, you are solving a Lasso
            penalty = L1(alpha)
        else:
            penalty = WeightedL1(alpha=alpha, weights=weights)
        super().__init__(
            datafit=Quadratic(), penalty=penalty, is_classif=False,
            max_iter=max_iter, max_epochs=max_epochs, p0=p0, tol=tol,
            fit_intercept=fit_intercept,
            warm_start=warm_start, ws_strategy=ws_strategy, verbose=verbose)
        self.alpha = alpha
        self.weights = weights

    def get_params(self, deep=False):
        """Get parameters of the estimators.

        Parameters
        ----------
        deep : bool
            Whether or not return the parameters for contained subobjects estimators.

        Returns
        -------
        params : dict
            The parameters of the estimator.
        """
        return super().get_params(deep)


class ElasticNet(GeneralizedLinearEstimator):
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
        Maximum number of iterations (subproblem definitions).

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    tol : float, optional
        Stopping criterion for the optimization.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    verbose : bool or int
        Amount of verbosity.

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

    def __init__(self, alpha=1., l1_ratio=0.5, max_iter=100,
                 max_epochs=50_000, p0=10, tol=1e-4, fit_intercept=False,
                 warm_start=False, verbose=0):
        super().__init__(
            datafit=Quadratic(), penalty=L1_plus_L2(alpha=alpha, l1_ratio=l1_ratio),
            is_classif=False, max_iter=max_iter, max_epochs=max_epochs, p0=p0, tol=tol,
            fit_intercept=fit_intercept,
            warm_start=warm_start, ws_strategy='subdiff', verbose=verbose)
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def get_params(self, deep=False):
        """Get parameters of the estimators.

        Parameters
        ----------
        deep : bool
            Whether or not return the parameters for contained subobjects estimators.

        Returns
        -------
        params : dict
            The parameters of the estimator.
        """
        return super().get_params(deep)


class MCPRegression(GeneralizedLinearEstimator):
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
        Maximum number of iterations (subproblem definitions).

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

    def __init__(self, alpha=1., gamma=3, max_iter=100, max_epochs=50_000, p0=10,
                 verbose=0, tol=1e-4, fit_intercept=False, warm_start=False):
        super().__init__(
            datafit=Quadratic(), penalty=MCPenalty(alpha=alpha, gamma=gamma),
            is_classif=False, max_iter=max_iter, max_epochs=max_epochs, p0=p0, tol=tol,
            fit_intercept=fit_intercept,
            warm_start=warm_start, ws_strategy='subdiff', verbose=verbose)
        self.gamma = gamma
        self.alpha = alpha

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
        return super().get_params(deep)


class SparseLogisticRegression(GeneralizedLinearEstimator):
    r"""Sparse Logistic regression estimator.

    The optimization objective for sparse Logistic regression is::

        mean(log(1 + exp(-y_i x_i^T w))) + alpha * ||w||_1

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength; must be a positive float.

    tol : float, optional
        Stopping criterion for the optimization: the solver runs until the
        duality gap is smaller than ``tol * len(y) * log(2)`` or the
        maximum number of iteration is reached.

    fit_intercept : bool, optional (default=False)
        Whether or not to fit an intercept. Currently True is not supported.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions).

    verbose : bool or int
        Amount of verbosity.

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        Only False is supported so far.

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

    def __init__(
            self, alpha=1., tol=1e-4,
            fit_intercept=False, max_iter=50, verbose=0,
            max_epochs=50000, p0=10, warm_start=False):

        super().__init__(
            datafit=Logistic(), penalty=L1(alpha=alpha),
            is_classif=True, max_iter=max_iter, max_epochs=max_epochs, p0=p0, tol=tol,
            fit_intercept=fit_intercept,
            warm_start=warm_start, ws_strategy='subdiff', verbose=verbose)
        self.alpha = alpha

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
        return super().get_params(deep)


class LinearSVC(LinearSVC_sklearn):
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

    tol : float, optional
        Stopping criterion for the optimization.

    fit_intercept : bool, optional
        Whether or not to fit an intercept. Currently True is not supported.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    verbose : bool or int
        Amount of verbosity.

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

    def __init__(
            self, C=1., max_iter=100, max_epochs=50_000, p0=10, tol=1e-4,
            fit_intercept=False, warm_start=False, verbose=0):

        super(LinearSVC, self).__init__(
            C=C, tol=tol, max_iter=max_iter, penalty='l2', loss='hinge',
            fit_intercept=False)
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.p0 = p0
        self.C = C
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept

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
        self.intercept_ = 0.

        if self.fit_intercept:
            raise NotImplementedError(
                "Fitting an intercept is not implement yet")

        if self.C < 0:
            raise ValueError(
                "Penalty term must be positive; got (C=%r)" % self.C)

        X, y = self._validate_data(
            X, y, accept_sparse="csc", dtype=np.float64,
            order="C", accept_large_sparse=True)
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        if not hasattr(self, "n_features_in_"):
            self.n_features_in_ = X.shape[1]

        enc = LabelEncoder()
        y_ind = enc.fit_transform(y)
        self.classes_ = enc.classes_
        n_classes = len(enc.classes_)

        if n_classes <= 2:
            y_ind = 2 * y_ind - 1
            is_sparse = issparse(X)
            if is_sparse:
                yXT = (X.T).multiply(y_ind)
            else:
                yXT = (X * y_ind[:, None]).T

            # Dirty hack to force casting, could find something better
            yXT = yXT.astype(X.dtype)
            _, coefs, _, self.n_iter_ = self.path(
                yXT, y_ind, np.array([self.C]))
            self.coef_ = coefs.T  # must be [1, n_features]
            self.intercept_ = 0
            if is_sparse:
                primal_coef = ((yXT).multiply(self.coef_[0, :])).T
            else:
                primal_coef = (yXT * self.coef_[0, :]).T

            # Primal-Dual relation
            primal_coef = primal_coef.sum(axis=0)
            self.dual_ = self.coef_
            # must be [1, n_features]
            self.coef_ = np.array(primal_coef).reshape(1, -1)
        else:
            self.coef_ = np.empty([n_classes, X.shape[1]])
            self.intercept_ = 0.
            multiclass = OneVsRestClassifier(self).fit(X, y)
            self.coef_ = np.array([clf.coef_[0]
                                   for clf in multiclass.estimators_])
            self.n_iter_ = max(clf.n_iter_ for clf in multiclass.estimators_)
        return self

    def path(self, yXT, y, Cs, coef_init=None, return_n_iter=True, **params):
        """Compute linear SVC path.

        Parameters
        ----------
        yXT : {array-like, sparse matrix} of shape (n_features, n_samples)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Cs : ndarray shape (n_Cs,)
            Values of regularization strenghts for which solutions are
            computed.

        coef_init : array, shape (n_features,), optional
            Initial value of the coefficients.

        return_n_iter : bool, optional (default=True)
            Whether or not to return the number of iterations.

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
        penalty_dual = compiled_clone(IndicatorBox(self.C))
        datafit = compiled_clone(QuadraticSVC(), to_float32=yXT.dtype == np.float32)

        return cd_solver_path(
            yXT, y, datafit, penalty_dual, alphas=Cs,
            fit_intercept=self.fit_intercept, coef_init=coef_init,
            max_iter=self.max_iter, return_n_iter=return_n_iter,
            max_epochs=self.max_epochs, p0=self.p0, tol=self.tol,
            verbose=self.verbose)


class MultiTaskLasso(MultiTaskLasso_sklearn):
    r"""MultiTaskLasso estimator.

    The optimization objective for MultiTaskLasso is::

        (1 / (2 * n_samples)) * ||y - X W||^2_2 + alpha * ||W||_{21}

    Parameters
    ----------
    alpha : float, optional
        Regularization strength (constant that multiplies the L21 penalty).

    max_iter : int, optional
        Maximum number of iterations (subproblem definitions).

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or int
        Amount of verbosity.

    tol : float, optional
        Stopping criterion for the optimization: the solver runs until the
        duality gap is smaller than ``tol * norm(y) ** 2 / len(y)`` or the
        maximum number of iteration is reached.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

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

    def __init__(self, alpha=1., max_iter=100,
                 max_epochs=50000, p0=10, verbose=0, tol=1e-4,
                 fit_intercept=False, warm_start=False):
        super().__init__(
            alpha=alpha, tol=tol, max_iter=max_iter,
            fit_intercept=fit_intercept, warm_start=warm_start)
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.p0 = p0
        self.datafit = QuadraticMultiTask()
        self.penalty = L2_1(alpha)

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
        # TODO check if we could just patch `bcd_solver_path` as we do in Lasso case.
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

        X, Y, X_offset, Y_offset, X_scale = _preprocess_data(
            X, Y, self.fit_intercept, copy=False)

        if not self.warm_start or not hasattr(self, "coef_"):
            self.coef_ = None

        _, coefs, kkt = self.path(
            X, Y, alphas=[self.alpha],
            coef_init=self.coef_, max_iter=self.max_iter,
            max_epochs=self.max_epochs, p0=self.p0, verbose=self.verbose,
            tol=self.tol)

        self.coef_, self.dual_gap_ = coefs[..., 0], kkt[-1]
        self.n_iter_ = len(kkt)
        self._set_intercept(X_offset, Y_offset, X_scale)

        return self

    def path(self, X, Y, alphas, coef_init=None, **params):
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
        datafit = compiled_clone(self.datafit, to_float32=X.dtype == np.float32)
        penalty = compiled_clone(self.penalty)

        return bcd_solver_path(X, Y, datafit, penalty, alphas=alphas,
                               coef_init=coef_init, **params)
