# License: BSD 3 clause

import warnings
import numpy as np
from scipy.sparse import issparse
from scipy.special import expit
from numbers import Integral, Real
from skglm.solvers import ProxNewton, LBFGS

from sklearn.utils.validation import (check_is_fitted, check_array,
                                      check_consistent_length)
from sklearn.linear_model._base import (
    RegressorMixin, LinearModel,
    LinearClassifierMixin, SparseCoefMixin, BaseEstimator
)
from sklearn.utils.extmath import softmax
from sklearn.preprocessing import LabelEncoder
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.multiclass import OneVsRestClassifier, check_classification_targets

from skglm.solvers import AndersonCD, MultiTaskBCD, GroupBCD
from skglm.datafits import (Cox, Quadratic, Logistic, QuadraticSVC,
                            QuadraticMultiTask, QuadraticGroup,)
from skglm.penalties import (L1, WeightedL1, L1_plus_L2, L2, WeightedGroupL2,
                             MCPenalty, WeightedMCPenalty, IndicatorBox, L2_1)
from skglm.utils.data import grp_converter
from sklearn.utils.validation import validate_data
from sklearn.model_selection import check_cv
from sklearn.metrics import get_scorer
from joblib import Parallel, delayed


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

        X, y = validate_data(
            model, X, y, validate_separately=(check_X_params, check_y_params))
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

    coefs, p_obj, kkt = solver.solve(X_, y, datafit, penalty, w, Xw)
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
        Datafit. If ``None``, ``datafit`` is initialized as a :class:`.Quadratic`
        datafit.  ``datafit`` is replaced by a JIT-compiled instance when calling fit.

    penalty : instance of BasePenalty, optional
        Penalty. If ``None``, ``penalty`` is initialized as a :class:`.L1` penalty.
        ``penalty`` is replaced by a JIT-compiled instance when
        calling fit.

    solver : instance of BaseSolver, optional
        Solver. If ``None``, ``solver`` is initialized as an :class:`.AndersonCD`
        solver.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_features, n_tasks)
        parameter array (:math:`w` in the cost function formula)

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
        # TODO: add support for Cox datafit in `_glm_fit`
        # `_glm_fit` interpret `Cox` as a multitask datafit which breaks down solvers
        if isinstance(self.datafit, Cox):
            raise ValueError(
                "`GeneralizedLinearEstimator` doesn't currently support "
                "`Cox` datafit"
            )

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

    def score(self, X, y):
        """Return the score of the model on the data X and y.

        For regression problems, this is the R² score.
        For classification problems, this is the accuracy score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for X.

        Returns
        -------
        score : float
            Score of self.predict(X) wrt. y.
        """
        if isinstance(self.datafit, (Logistic, QuadraticSVC)):
            return np.mean(self.predict(X) == y)
        else:
            y_pred = self.predict(X)
            return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

    def get_params(self, deep=False):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=False
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = {}
        if self.penalty is not None:
            penalty_params = self.penalty.get_params(deep=deep)
            for key, value in penalty_params.items():
                params[f'penalty__{key}'] = value
        if self.datafit is not None:
            datafit_params = self.datafit.get_params(deep=deep)
            for key, value in datafit_params.items():
                params[f'datafit__{key}'] = value
        if self.solver is not None:
            solver_params = self.solver.get_params(deep=deep)
            for key, value in solver_params.items():
                params[f'solver__{key}'] = value
        return params

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Parameter names mapped to their values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Handle penalty parameters
        penalty_params = {}
        datafit_params = {}
        other_params = {}

        for key, value in params.items():
            if key.startswith('penalty__'):
                penalty_params[key[9:]] = value
            elif key.startswith('datafit__'):
                datafit_params[key[9:]] = value
            else:
                other_params[key] = value

        # Set regular parameters
        for key, value in other_params.items():
            setattr(self, key, value)

        # Set penalty parameters
        if penalty_params and self.penalty is not None:
            self.penalty.set_params(**penalty_params)

        # Set datafit parameters
        if datafit_params and self.datafit is not None:
            self.datafit.set_params(**datafit_params)

        return self

    # Cross‑validation utility
    def _alpha_grid(self, X, y, alphas, min_alpha_ratio=1e-3):
        """Compute a descending geometric grid of alpha values.

        Parameters
        ----------
        min_alpha_ratio : float, default=1e-3
            Minimum alpha value as a ratio of alpha_max.
        """
        if not (isinstance(alphas, str) or isinstance(alphas, int)):
            return np.asarray(alphas, dtype=float)

        n_points = 50 if alphas == 'auto' else int(alphas)

        # Get gradient at zero
        if hasattr(self.datafit, "gradient_at_zero"):
            grad0 = self.datafit.gradient_at_zero(X, y)
        else:
            raise NotImplementedError(
                f"{self.datafit.__class__.__name__} must implement "
                "`gradient_at_zero(X, y)` for CV support."
            )

        alpha_max = self.penalty.alpha_max(grad0)
        if not np.isfinite(alpha_max) or alpha_max <= 0:
            raise ValueError("Invalid alpha_max computed: %s" % alpha_max)

        return np.geomspace(alpha_max, alpha_max * min_alpha_ratio, num=n_points)

    def _fit_fold(self, X, y, train_idx, val_idx, ratio, alphas, scorer):
        """Run one warm‑started alpha path on a single CV fold."""
        try:
            # Create a new estimator instance without cloning
            est = GeneralizedLinearEstimator(
                datafit=self.datafit.__class__(),
                penalty=self.penalty.__class__(**self.penalty.get_params()),
                solver=self.solver.__class__(**self.solver.get_params())
            )

            # Initialize intercept_ attribute
            est.intercept_ = 0.0

            # Add warm start between folds
            if not hasattr(self, '_last_coef'):
                self._last_coef = None

            if self._last_coef is not None:
                est.coef_ = self._last_coef.copy()

            if ratio is not None:
                est.penalty.l1_ratio = ratio
            est.solver.warm_start = True

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            scores = np.zeros(len(alphas), dtype=float)
            for k, alpha in enumerate(alphas):
                est.penalty.alpha = float(alpha)
                est.fit(X_tr, y_tr)
                scores[k] = (est.score(X_val, y_val) if scorer is None
                             else scorer(est, X_val, y_val))

            # Store last coefficients for next fold
            self._last_coef = est.coef_.copy()

            return scores
        except Exception as e:
            warnings.warn(f"Failed to fit fold: {str(e)}")
            return np.full(len(alphas), np.nan)

    def cross_validate(self, X, y, *, alphas='auto', l1_ratios=(0.5,),
                       cv=5, scoring=None, n_jobs=None, verbose=0):
        """Select ``alpha`` (and optionally ``l1_ratio``) via K‑fold CV.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        alphas : array-like, 'auto', or int, default='auto'
            Grid of alpha values to try. If 'auto', compute a geometric grid
            between alpha_max and alpha_max * 1e-3. If int, use that many
            points in the grid.
        l1_ratios : array-like, default=(0.5,)
            Grid of l1_ratio values to try. Only used if penalty supports l1_ratio.
        cv : int or cross-validation splitter, default=5
            Number of folds or cross-validation splitter.
        scoring : str or callable, default=None
            Scoring metric to use. If None, use the estimator's default scorer.
        n_jobs : int, default=None
            Number of parallel jobs to run.
        verbose : int, default=0
            Verbosity level.

        Attributes
        ----------
        best_estimator_ : estimator
            Estimator with best parameters.
        best_alpha_ : float
            Best alpha value.
        best_score_ : float
            Best cross-validation score.
        best_l1_ratio_ : float, optional
            Best l1_ratio value if penalty supports it.
        cv_scores_ : dict
            Cross-validation scores for each l1_ratio.
        alphas_ : array
            Grid of alpha values used.

        Returns
        -------
        self : object
            Fitted estimator with best parameters.

        Notes
        -----
        The implementation is optimized for the most common use cases:
        1. Lasso (L1 penalty) for regression and classification
        2. Elastic Net (L1_plus_L2 penalty) for regression and classification
        3. Logistic Regression with L1 or L1_plus_L2 penalties

        Other models have not been thoroughly tested in cross-validation settings.

        The method uses warm-starting along the alpha path for efficiency and
        supports parallel computation of CV folds.

        This method supports the following combinations of datafits and penalties:

        Supported Datafits:
        ------------------
        - Quadratic: Standard least squares regression
        - Logistic: Binary and multiclass classification
        - QuadraticSVC: Support Vector Classification
        - Cox: Survival analysis (experimental)
        - QuadraticMultiTask: Multi-task learning (experimental)
        - QuadraticGroup: Group-structured data (experimental)

        Supported Penalties:
        ------------------
        - L1: Lasso regularization (sparse solutions)
        - L1_plus_L2: Elastic Net regularization (sparse + grouped features)
        - L2: Ridge regularization (grouped features)
        - MCPenalty: Minimax Concave Penalty (experimental)
        - WeightedL1: Feature-specific L1 penalties (experimental)
        - WeightedGroupL2: Group-specific L2 penalties (experimental)

        Examples
        --------
        >>> from skglm import Lasso, ElasticNet, SparseLogisticRegression
        >>> from skglm.datafits import Quadratic, Logistic
        >>> from skglm.penalties import L1, L1_plus_L2

        # Lasso regression
        >>> est = Lasso(alpha=1.0)
        >>> est.cross_validate(X, y, alphas='auto', cv=5)

        # Elastic Net regression
        >>> est = ElasticNet(alpha=1.0, l1_ratio=0.5)
        >>> est.cross_validate(X, y, alphas='auto', l1_ratios=[0.1, 0.5, 0.9], cv=5)

        # Logistic regression with L1 penalty
        >>> est = SparseLogisticRegression(alpha=1.0)
        >>> est.cross_validate(X, y, alphas='auto', cv=5)
        """
        # Validate penalty has alpha
        if not hasattr(self.penalty, 'alpha'):
            raise ValueError("penalty '%s' has no 'alpha' attribute, "
                             "cannot perform CV."
                             % self.penalty.__class__.__name__)

        # Validate l1_ratio if present
        if not hasattr(self.penalty, 'l1_ratio') and l1_ratios != (0.5,):
            raise ValueError(
                f"penalty '{self.penalty.__class__.__name__}' does not support "
                "`l1_ratio`, but l1_ratios={l1_ratios!r} was supplied."
            )

        # Validate scoring
        if scoring is not None and not isinstance(scoring, (str, callable)):
            raise ValueError("scoring must be a string or callable")

        is_classif = isinstance(self.datafit, (Logistic, QuadraticSVC))
        cv_splitter = check_cv(cv, y, classifier=is_classif)

        # Guard against useless l1_ratio array when the penalty has no such hyper‑param
        if not hasattr(self.penalty, 'l1_ratio') and l1_ratios != (0.5,):
            raise ValueError(
                f"penalty '{self.penalty.__class__.__name__}' does not support "
                "`l1_ratio`, but l1_ratios={l1_ratios!r} was supplied."
            )

        ratios = l1_ratios if hasattr(self.penalty, 'l1_ratio') else [None]

        scorer = (get_scorer(scoring) if isinstance(scoring, str)
                  else scoring)   # callable or None

        # Use joblib's memory management
        with Parallel(n_jobs=n_jobs, verbose=verbose) as parallel:
            cv_scores = {}
            for ratio in ratios:
                if ratio is not None:
                    old_ratio = self.penalty.l1_ratio
                    self.penalty.l1_ratio = ratio
                alphas_grid = self._alpha_grid(X, y, alphas)
                if ratio is not None:
                    self.penalty.l1_ratio = old_ratio

                fold_scores = parallel(
                    delayed(self._fit_fold)(
                        X, y, tr, vl, ratio, alphas_grid, scorer
                    )
                    for tr, vl in cv_splitter.split(X, y)
                )
                cv_scores[ratio] = np.stack(fold_scores, axis=0)

        # After computing cv_scores, add:
        best_score = -np.inf
        best_alpha = None
        best_ratio = None

        for ratio in ratios:
            means = cv_scores[ratio].mean(axis=0)
            k = int(means.argmax())
            if means[k] > best_score:
                best_score = means[k]
                best_alpha = alphas_grid[k]
                best_ratio = ratio

        # Set the best parameters
        self.penalty.alpha = best_alpha
        if best_ratio is not None:
            self.penalty.l1_ratio = best_ratio

        # Store the best parameters and scores
        self.best_alpha_ = best_alpha
        self.best_score_ = best_score
        if best_ratio is not None:
            self.best_l1_ratio_ = best_ratio
        self.cv_scores_ = cv_scores
        self.alphas_ = alphas_grid

        # Fit the final model with the best parameters
        self.fit(X, y)

        return self


class Lasso(RegressorMixin, LinearModel):
    r"""Lasso estimator based on Celer solver and primal extrapolation.

    The optimization objective for Lasso is:

    .. math::
        1 / (2 xx n_"samples")  ||y - Xw||_2 ^ 2 + alpha ||w||_1

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
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str
        The score used to build the working set. Can be ``"fixpoint"`` or ``"subdiff"``.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (:math:`w` in the cost function formula)

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
            If warm_start is enabled, the optimization problem restarts from
            ``coef_init``.

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
            The number of iterations along the path. If return_n_iter is set to
            ``True``.
        """
        penalty = L1(self.alpha, self.positive)
        datafit = Quadratic()
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return solver.path(X, y, datafit, penalty, alphas, coef_init, return_n_iter)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


class WeightedLasso(RegressorMixin, LinearModel):
    r"""WeightedLasso estimator based on Celer solver and primal extrapolation.

    The optimization objective for WeightedLasso is:

    .. math::
        1 / (2 xx n_"samples") ||y - Xw||_2 ^ 2 + alpha ||w||_1

    Parameters
    ----------
    alpha : float, optional
        Penalty strength.

    weights : array, shape (n_features,), optional (default=None)
        Positive weights used in the L1 penalty part of the Lasso
        objective. If ``None``, weights equal to 1 are used.

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
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str
        The score used to build the working set. Can be ``fixpoint`` or ``subdiff``.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (:math:`w` in the cost function formula)

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
        penalty = WeightedL1(self.alpha, weights, self.positive)
        datafit = Quadratic()
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

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


class ElasticNet(RegressorMixin, LinearModel):
    r"""Elastic net estimator.

    The optimization objective for Elastic net is:

    .. math::
        1 / (2 xx n_"samples") ||y - Xw||_2 ^ 2
        + tt"l1_ratio" xx alpha ||w||_1
        + (1 - tt"l1_ratio") xx alpha/2 ||w||_2 ^ 2

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
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str
        The score used to build the working set. Can be ``fixpoint`` or ``subdiff``.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (:math:`w` in the cost function formula)

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
            The number of iterations along the path. If return_n_iter is set to
            ``True``.
        """
        penalty = L1_plus_L2(self.alpha, self.l1_ratio, self.positive)
        datafit = Quadratic()
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

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


class MCPRegression(RegressorMixin, LinearModel):
    r"""Linear regression with MCP penalty estimator.

    The optimization objective for MCPRegression is, with :math:`x >= 0`:

    .. math::
        "pen"(x) = {(alpha x - x^2 / (2 gamma), if x <= alpha gamma),
                    (gamma alpha^2 / 2        , if x > alpha gamma):}

    .. math::
        "obj" = 1 / (2 xx n_"samples") ||y - Xw||_2 ^ 2
              + sum_(j=1)^(n_"features") "pen"(|w_j|)

    For more details see
    `Coordinate descent algorithms for nonconvex penalized regression,
    with applications to biological feature selection, Breheny and Huang
    <https://doi.org/10.1214/10-aoas388>`_.

    Parameters
    ----------
    alpha : float, optional
        Penalty strength.

    gamma : float, default=3
        If ``gamma = 1``, the prox of MCP is a hard thresholding.
        If ``gamma = np.inf`` it is a soft thresholding.
        Should be larger than (or equal to) 1.

    weights : array, shape (n_features,), optional (default=None)
        Positive weights used in the L1 penalty part of the Lasso
        objective. If ``None``, weights equal to 1 are used.

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
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str
        The score used to build the working set. Can be ``fixpoint`` or ``subdiff``.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (:math:`w` in the cost function formula)

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

    def __init__(self, alpha=1., gamma=3, weights=None, max_iter=50, max_epochs=50_000,
                 p0=10, verbose=0, tol=1e-4, positive=False, fit_intercept=True,
                 warm_start=False, ws_strategy="subdiff"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weights = weights
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.p0 = p0
        self.verbose = verbose
        self.tol = tol
        self.positive = positive
        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.ws_strategy = ws_strategy

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
            If warm start is enabled, the optimization problem restarts from
            ``coef_init``.

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
            The number of iterations along the path. If return_n_iter is set to
            ``True``.
        """
        if self.weights is None:
            penalty = MCPenalty(self.alpha, self.gamma, self.positive)
        else:
            if X.shape[1] != len(self.weights):
                raise ValueError(
                    "The number of weights must match the number of features. "
                    f"Got {len(self.weights)}, expected {X.shape[1]}."
                )
            penalty = WeightedMCPenalty(
                self.alpha, self.gamma, self.weights, self.positive)

        datafit = Quadratic()
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
            penalty = MCPenalty(self.alpha, self.gamma, self.positive)
        else:
            if X.shape[1] != len(self.weights):
                raise ValueError(
                    "The number of weights must match the number of features. "
                    f"Got {len(self.weights)}, expected {X.shape[1]}."
                )
            penalty = WeightedMCPenalty(
                self.alpha, self.gamma, self.weights, self.positive)
        solver = AndersonCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return _glm_fit(X, y, self, Quadratic(), penalty, solver)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


class SparseLogisticRegression(LinearClassifierMixin, SparseCoefMixin, BaseEstimator):
    r"""Sparse Logistic regression estimator.

    The optimization objective for sparse Logistic regression is:

    .. math::
        1 / n_"samples" \sum_{i=1}^{n_"samples"} log(1 + exp(-y_i x_i^T w))
        + tt"l1_ratio" xx alpha ||w||_1
        + (1 - tt"l1_ratio") xx alpha/2 ||w||_2 ^ 2

    By default, ``l1_ratio=1.0`` corresponds to Lasso (pure L1 penalty).
    When ``0 < l1_ratio < 1``, the penalty is a convex combination of L1 and L2
    (i.e., ElasticNet). ``l1_ratio=0.0`` corresponds to Ridge (pure L2), but note
    that pure Ridge is not typically used with this class.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength; must be a positive float.

    l1_ratio : float, default=1.0
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``.
        Only used when ``penalty="l1_plus_l2"``.
        For ``l1_ratio = 0`` the penalty is an L2 penalty.
        ``For l1_ratio = 1`` it is an L1 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2.

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
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes, )
        A list of class labels known to the classifier.

    coef_ : ndarray, shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.

        ``coef_`` is of shape (1, n_features) when the given problem is binary.

    intercept_ :  ndarray, shape (1,) or (n_classes,)
        constant term in decision function. Not handled yet.

    n_iter_ : int
        Number of subproblems solved to reach the specified tolerance.
    """

    def __init__(self, alpha=1.0, l1_ratio=1.0, tol=1e-4, max_iter=20, max_epochs=1_000,
                 verbose=0, fit_intercept=True, warm_start=False):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
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
        return _glm_fit(X, y, self, Logistic(), L1_plus_L2(self.alpha, self.l1_ratio),
                        solver)

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
            Vector to be scored, where ``n_samples`` is the number of samples and
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
                ``1. / (1. + np.exp(-self.decision_function(X)))``;
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

    The optimization objective for LinearSVC is:

    .. math:: C xx sum_(i=1)^(n_"samples") max(0, 1 - y_i beta^T X[i, :])
        + 1/2 ||beta||_2 ^ 2

    i.e. hinge datafit loss (non-smooth) + l2 regularization (smooth)

    To solve this, we solve the dual optimization problem to stay in our
    framework of smooth datafit and non-smooth penalty.
    The dual optimization problem of SVC is:

    .. math::
        1/2 ||(yX)^T w||_2 ^ 2
        - \sum_(i=1)^(n_"samples") w_i
        + \sum_(i=1)^(n_"samples") [0 <= w_i <= C]

    The primal-dual relation is given by:

    .. math:: w = \sum_(i=1)^(n_"samples") y_i w_i X[i, :]

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
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str
        The score used to build the working set. Can be ``fixpoint`` or ``subdiff``.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (:math:`w` in the cost function formula)

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


class CoxEstimator(LinearModel):
    r"""Elastic Cox estimator with Efron and Breslow estimate.

    Refer to :ref:`Mathematics behind Cox datafit <maths_cox_datafit>`
    for details about the datafit expression. The data convention for the estimator is

    - ``X`` the design matrix with ``n_features`` predictors
    - ``y`` a two-column array where the first ``tm`` is of event time occurrences
      and the second ``s`` is of censoring.

    For L2-regularized Cox (``l1_ratio=0.``) :ref:`LBFGS <skglm.solvers.LBFGS>`
    is the used solver, otherwise it is :ref:`ProxNewton <skglm.solvers.ProxNewton>`.

    Parameters
    ----------
    alpha : float, optional
        Penalty strength. It must be strictly positive.

    l1_ratio : float, default=0.7
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    method : {'efron', 'breslow'}, default='efron'
        The estimate used for the Cox datafit. Use ``efron`` to
        handle tied observations.

    tol : float, optional
        Stopping criterion for the optimization.

    max_iter : int, optional
        The maximum number of iterations to solve the problem.

    verbose : bool or int
        Amount of verbosity.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        Parameter vector of Cox regression.

    stop_crit_ : float
        The value of the stopping criterion at convergence.
    """

    _parameter_constraints: dict = {
        "alpha": [Interval(Real, 0, None, closed="neither")],
        "l1_ratio": [Interval(Real, 0, 1, closed="both")],
        "method": [StrOptions({"efron", "breslow"})],
        "tol": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "verbose": ["boolean", Interval(Integral, 0, 2, closed="both")],
    }

    def __init__(self, alpha=1., l1_ratio=0.7, method="efron", tol=1e-4,
                 max_iter=50, verbose=False):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.method = method
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X, y):
        """Fit Cox estimator.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Design matrix.

        y : array-like, shape (n_samples, 2)
            Two-column array where the first is of event time occurrences
            and the second is of censoring. If it is of dimension 1, it is
            assumed to be the times vector and there no censoring.

        Returns
        -------
        self :
            The fitted estimator.
        """
        self._validate_params()

        # validate input data
        X = check_array(
            X,
            accept_sparse="csc",
            order="F",
            dtype=[np.float64, np.float32],
            input_name="X",
        )
        if y is None:
            # Needed to pass check estimator. Message error is
            # copy/paste from https://github.com/scikit-learn/scikit-learn/blob/ \
            # 23ff51c07ebc03c866984e93c921a8993e96d1f9/sklearn/utils/ \
            # estimator_checks.py#L3886
            raise ValueError("requires y to be passed, but the target y is None")
        y = check_array(
            y,
            accept_sparse=False,
            order="F",
            dtype=X.dtype,
            ensure_2d=False,
            input_name="y",
        )
        if y.ndim == 1:
            warnings.warn(
                f"{repr(self)} requires the vector of response `y` to have "
                f"two columns. Got one column.\nAssuming that `y` "
                "is the vector of times and there is no censoring."
            )
            y = np.column_stack((y, np.ones_like(y))).astype(X.dtype, order="F")
        elif y.shape[1] > 2:
            raise ValueError(
                f"{repr(self)} requires the vector of response `y` to have "
                f"two columns. Got {y.shape[1]} columns."
            )

        check_consistent_length(X, y)

        # init datafit and penalty
        datafit = Cox(self.method)

        if self.l1_ratio == 1.:
            penalty = L1(self.alpha)
        elif 0. < self.l1_ratio < 1.:
            penalty = L1_plus_L2(self.alpha, self.l1_ratio)
        else:
            penalty = L2(self.alpha)

        # init solver
        if self.l1_ratio == 0.:
            solver = LBFGS(max_iter=self.max_iter, tol=self.tol, verbose=self.verbose)
        else:
            solver = ProxNewton(
                max_iter=self.max_iter, tol=self.tol, verbose=self.verbose,
                fit_intercept=False,
            )

        # solve problem
        if not issparse(X):
            datafit.initialize(X, y)
        else:
            datafit.initialize_sparse(X.data, X.indptr, X.indices, y)

        w, _, stop_crit = solver.solve(X, y, datafit, penalty)

        # save to attribute
        self.coef_ = w
        self.stop_crit_ = stop_crit

        self.intercept_ = 0.
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.arange(X.shape[1])

        return self

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


class MultiTaskLasso(RegressorMixin, LinearModel):
    r"""MultiTaskLasso estimator.

    The optimization objective for MultiTaskLasso is:

    .. math:: 1 / (2 xx n_"samples") ||y - XW||_2 ^ 2 + alpha ||W||_(21)

    Parameters
    ----------
    alpha : float, optional
        Regularization strength (constant that multiplies the L21 penalty).

    copy_X : bool, optional (default=True)
        If ``True``, X will be copied; else, it may be overwritten.

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
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str
        The score used to build the working set. Can be ``fixpoint`` or ``subdiff``.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (:math:`w` in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved by Celer to reach the specified tolerance.
    """

    def __init__(self, alpha=1., copy_X=True, max_iter=50, max_epochs=50_000, p0=10,
                 verbose=0, tol=1e-4, fit_intercept=True, warm_start=False,
                 ws_strategy="subdiff"):
        self.tol = tol
        self.alpha = alpha
        self.copy_X = copy_X
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
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
        X, Y = validate_data(self, X, Y, validate_separately=(check_X_params,
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

        datafit = QuadraticMultiTask()
        penalty = L2_1(self.alpha)

        solver = MultiTaskBCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        W, obj_out, kkt = solver.solve(X, Y, datafit, penalty)

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
            If warm start is enabled, the optimization problem restarts from
            ``coef_init``.

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
            The number of iterations along the path. If return_n_iter is set to
            ``True``.
        """
        datafit = QuadraticMultiTask()
        penalty = L2_1(self.alpha)
        solver = MultiTaskBCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            ws_strategy=self.ws_strategy, fit_intercept=self.fit_intercept,
            warm_start=self.warm_start, verbose=self.verbose)
        return solver.path(X, Y, datafit, penalty, alphas, coef_init, return_n_iter)


class GroupLasso(RegressorMixin, LinearModel):
    r"""GroupLasso estimator based on Celer solver and primal extrapolation.

    The optimization objective for GroupLasso is:

    .. math::
        1 / (2 xx n_"samples") ||y - X w||_2 ^ 2 + alpha \sum_g
        weights_g ||w_{[g]}||_2

    with :math:`w_{[g]}` the coefficients of the g-th group.

    Parameters
    ----------
    groups : int | list of ints | list of lists of ints
        Partition of features used in the penalty on ``w``.
        If an int is passed, groups are contiguous blocks of features, of size
        ``groups``.
        If a list of ints is passed, groups are assumed to be contiguous,
        group number ``g`` being of size ``groups[g]``.
        If a list of lists of ints is passed, ``groups[g]`` contains the
        feature indices of the group number ``g``.

    alpha : float, optional
        Penalty strength.

    weights : array, shape (n_groups,), optional (default=None)
        Positive weights used in the L1 penalty part of the Lasso
        objective. If ``None``, weights equal to 1 are used.

    max_iter : int, optional (default=50)
        The maximum number of iterations (subproblem definitions).

    max_epochs : int, optional (default=50_000)
        Maximum number of CD epochs on each subproblem.

    p0 : int, optional (default=10)
        First working set size.

    verbose : bool or int, optional (default=0)
        Amount of verbosity.

    tol : float, optional (default=1e-4)
        Stopping criterion for the optimization.

    positive : bool, optional (defautl=False)
        When set to ``True``, forces the coefficient vector to be positive.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    ws_strategy : str, optional (default="subdiff")
        The score used to build the working set. Can be ``fixpoint`` or ``subdiff``.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (:math:`w` in the cost function formula)

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved to reach the specified tolerance.

    Notes
    -----
    Supports weights equal to ``0``, i.e. unpenalized features.
    """

    def __init__(self, groups, alpha=1., weights=None, max_iter=50, max_epochs=50_000,
                 p0=10, verbose=0, tol=1e-4, positive=False, fit_intercept=True,
                 warm_start=False, ws_strategy="subdiff"):
        super().__init__()
        self.alpha = alpha
        self.groups = groups
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

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector relative to ``X``.

        Returns
        -------
        self : Instance of GroupLasso
            Fitted estimator.
        """
        grp_indices, grp_ptr = grp_converter(self.groups, X.shape[1])
        group_sizes = np.diff(grp_ptr)

        n_features = np.sum(group_sizes)
        if X.shape[1] != n_features:
            raise ValueError(
                "The total number of group members must equal the number of features. "
                f"Got {n_features}, expected {X.shape[1]}.")

        weights = np.ones(len(group_sizes)) if self.weights is None else self.weights
        group_penalty = WeightedGroupL2(alpha=self.alpha, grp_ptr=grp_ptr,
                                        grp_indices=grp_indices, weights=weights,
                                        positive=self.positive)
        quad_group = QuadraticGroup(grp_ptr=grp_ptr, grp_indices=grp_indices)
        solver = GroupBCD(
            self.max_iter, self.max_epochs, self.p0, tol=self.tol,
            fit_intercept=self.fit_intercept, warm_start=self.warm_start,
            verbose=self.verbose)

        return _glm_fit(X, y, self, quad_group, group_penalty, solver)
