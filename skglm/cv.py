import numpy as np
from joblib import Parallel, delayed
from skglm.datafits import Logistic, QuadraticSVC
from skglm.estimators import GeneralizedLinearEstimator
from sklearn.model_selection import KFold, StratifiedKFold


class GeneralizedLinearEstimatorCV(GeneralizedLinearEstimator):
    """Cross-validated wrapper for GeneralizedLinearEstimator.

    This class performs cross-validated selection of the regularization parameter(s)
    for a generalized linear estimator, supporting both L1 and elastic-net penalties.

    Parameters
    ----------
    datafit : object
        Datafit (loss) function instance (e.g., Logistic, Quadratic).
    penalty : object
        Penalty instance with an 'alpha' parameter (and optionally 'l1_ratio').
    solver : object
        Solver instance to use for optimization.
    alphas : array-like of shape (n_alphas,), optional
        List of alpha values to try. If None, they are set automatically.
    l1_ratio : float or array-like, optional
        The ElasticNet mixing parameter(s), with 0 <= l1_ratio <= 1.
        Only used if the penalty supports 'l1_ratio'. If None, defaults to 1.0 (Lasso).
    cv : int, default=4
        Number of cross-validation folds.
    n_jobs : int, default=1
        Number of jobs to run in parallel for cross-validation.
    random_state : int or None, default=None
        Random seed for cross-validation splitting.
    eps : float, default=1e-3
        Ratio of minimum to maximum alpha if alphas are set automatically.
    n_alphas : int, default=100
        Number of alphas along the regularization path if alphas are set automatically.

    Attributes
    ----------
    alpha_ : float
        Best alpha found by cross-validation.
    l1_ratio_ : float or None
        Best l1_ratio found by cross-validation (if applicable).
    best_estimator_ : GeneralizedLinearEstimator
        Estimator fitted on the full data with the best parameters.
    coef_ : ndarray
        Coefficients of the fitted model.
    intercept_ : float or ndarray
        Intercept of the fitted model.
    alphas_ : ndarray
        Array of alphas used in the search.
    scores_path_ : ndarray
        Cross-validation scores for each parameter combination.
    n_iter_ : int or None
        Number of iterations run by the solver (if available).
    n_features_in_ : int or None
        Number of features seen during fit.
    feature_names_in_ : ndarray or None
        Names of features seen during fit.
    """

    def __init__(self, datafit, penalty, solver, alphas=None, l1_ratio=None,
                 cv=4, n_jobs=1, random_state=None,
                 eps=1e-3, n_alphas=100):
        super().__init__(datafit=datafit, penalty=penalty, solver=solver)
        self.alphas = alphas
        self.l1_ratio = l1_ratio
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.eps = eps
        self.n_alphas = n_alphas

    def _score(self, y_true, y_pred):
        """Compute the performance score (higher is better)."""
        if isinstance(self.datafit, (Logistic, QuadraticSVC)):
            return float(np.mean(y_true == y_pred))
        return -float(np.mean((y_true - y_pred) ** 2))

    def fit(self, X, y):
        """Fit the model using cross-validation."""
        if not hasattr(self.penalty, "alpha"):
            raise ValueError(
                "GeneralizedLinearEstimatorCV only supports penalties which "
                "expose an 'alpha' parameter."
            )
        n_samples, n_features = X.shape

        if self.alphas is not None:
            alphas = np.sort(self.alphas)[::-1]
        else:
            alpha_max = np.max(np.abs(X.T @ y)) / n_samples
            alphas = np.geomspace(
                alpha_max,
                alpha_max * self.eps,
                self.n_alphas
            )
        has_l1_ratio = hasattr(self.penalty, "l1_ratio")
        l1_ratios = [1.] if not has_l1_ratio else np.atleast_1d(
            self.l1_ratio if self.l1_ratio is not None else [1.])

        scores_path = np.empty((len(l1_ratios), len(alphas), self.cv))
        best_loss = -np.inf

        def _solve_fold(k, train, test, alpha, l1, w_init):
            pen_kwargs = {k: v for k, v in self.penalty.__dict__.items()
                          if k not in ("alpha", "l1_ratio")}
            if has_l1_ratio:
                pen_kwargs['l1_ratio'] = l1
            pen = type(self.penalty)(alpha=alpha, **pen_kwargs)

            est = GeneralizedLinearEstimator(
                datafit=self.datafit, penalty=pen, solver=self.solver
            )
            if w_init is not None:
                est.coef_ = w_init[0]
                est.intercept_ = w_init[1]
            est.fit(X[train], y[train])
            y_pred = est.predict(X[test])
            return est.coef_, est.intercept_, self._score(y[test], y_pred)

        for idx_ratio, l1_ratio in enumerate(l1_ratios):
            warm_start = [None] * self.cv

            for idx_alpha, alpha in enumerate(alphas):
                if isinstance(self.datafit, (Logistic, QuadraticSVC)):
                    kf = StratifiedKFold(n_splits=self.cv, shuffle=True,
                                         random_state=self.random_state)
                    split_iter = kf.split(np.arange(n_samples), y)
                else:
                    kf = KFold(n_splits=self.cv, shuffle=True,
                               random_state=self.random_state)
                    split_iter = kf.split(np.arange(n_samples))
                fold_results = Parallel(self.n_jobs)(
                    delayed(_solve_fold)(k, tr, te, alpha, l1_ratio, warm_start[k])
                    for k, (tr, te) in enumerate(split_iter)
                )

                for k, (coef_fold, intercept_fold, loss_fold) in \
                        enumerate(fold_results):
                    warm_start[k] = (coef_fold, intercept_fold)
                    scores_path[idx_ratio, idx_alpha, k] = loss_fold

                mean_loss = np.mean(scores_path[idx_ratio, idx_alpha])
                if mean_loss > best_loss:
                    best_loss = mean_loss
                    self.alpha_ = float(alpha)
                    self.l1_ratio_ = float(l1_ratio) if has_l1_ratio else None

        # Refit on full dataset
        pen_kwargs = {k: v for k, v in self.penalty.__dict__.items()
                      if k not in ("alpha", "l1_ratio")}
        if has_l1_ratio:
            pen_kwargs["l1_ratio"] = self.l1_ratio_
        best_penalty = type(self.penalty)(
            alpha=self.alpha_, **pen_kwargs
        )
        best_estimator = GeneralizedLinearEstimator(
            datafit=self.datafit,
            penalty=best_penalty,
            solver=self.solver
        )
        best_estimator.fit(X, y)
        self.best_estimator_ = best_estimator
        self.coef_ = best_estimator.coef_
        self.intercept_ = best_estimator.intercept_
        self.n_iter_ = getattr(best_estimator, "n_iter_", None)
        self.n_features_in_ = getattr(best_estimator, "n_features_in_", None)
        self.feature_names_in_ = getattr(best_estimator, "feature_names_in_", None)
        self.alphas_ = alphas
        self.scores_path_ = np.squeeze(scores_path)
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def score(self, X, y):
        return self.best_estimator_.score(X, y)
