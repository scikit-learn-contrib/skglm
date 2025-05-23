import numpy as np
from joblib import Parallel, delayed
from sklearn.utils.extmath import softmax
from skglm.datafits import Logistic, QuadraticSVC
from skglm.estimators import GeneralizedLinearEstimator


def _kfold_split(n_samples, k, rng):
    indices = rng.permutation(n_samples)
    fold_size = n_samples // k
    extra = n_samples % k

    start = 0
    for i in range(k):
        end = start + fold_size + (1 if i < extra else 0)
        test = indices[start:end]
        train = np.concatenate([indices[:start], indices[end:]])
        yield train, test
        start = end


class GeneralizedLinearEstimatorCV(GeneralizedLinearEstimator):
    """CV wrapper for GeneralizedLinearEstimator."""

    def __init__(self, datafit, penalty, solver, alphas=None, l1_ratio=None,
                 cv=4, n_jobs=1, random_state=None, scoring=None,
                 eps=1e-3, n_alphas=100):
        super().__init__(datafit=datafit, penalty=penalty, solver=solver)
        self.alphas = alphas
        self.l1_ratio = l1_ratio
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.scoring = scoring
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
                "GeneralizedLinearEstimatorCV only supports penalties with 'alpha'."
            )
        y = np.asarray(y)
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        if self.alphas is not None:
            alphas = np.sort(self.alphas)[::-1]
        else:
            alpha_max = np.max(np.abs(X.T @ y)) / n_samples
            alphas = np.geomspace(
                alpha_max,
                alpha_max * self.eps,
                self.n_alphas
            )[::-1]
        has_l1_ratio = hasattr(self.penalty, "l1_ratio")
        l1_ratios = [1.] if not has_l1_ratio else np.atleast_1d(
            self.l1_ratio if self.l1_ratio is not None else [1.])

        mse_path = np.empty((len(l1_ratios), len(alphas), self.cv))
        best_loss = np.inf

        def _solve_fold(k, train, test, alpha, l1, w_start):
            pen_kwargs = {k: v for k, v in self.penalty.__dict__.items()
                          if k not in ("alpha", "l1_ratio")}
            if has_l1_ratio:
                pen_kwargs['l1_ratio'] = l1
            pen = type(self.penalty)(alpha=alpha, **pen_kwargs)

            kw = dict(X=X[train], y=y[train], datafit=self.datafit, penalty=pen)
            if 'w' in self.solver.solve.__code__.co_varnames:
                kw['w'] = w_start
            w = self.solver.solve(**kw)
            w = w[0] if isinstance(w, tuple) else w

            coef, intercept = (w[:n_features], w[n_features]
                               ) if w.size == n_features + 1 else (w, 0.0)

            y_pred = X[test] @ coef + intercept
            return w, self._score(y[test], y_pred)

        for idx_ratio, l1_ratio in enumerate(l1_ratios):
            warm_start = [None] * self.cv

            for idx_alpha, alpha in enumerate(alphas):
                fold_results = Parallel(self.n_jobs)(
                    delayed(_solve_fold)(k, tr, te, alpha, l1_ratio, warm_start[k])
                    for k, (tr, te) in enumerate(_kfold_split(n_samples, self.cv, rng))
                )

                for k, (w_fold, loss_fold) in enumerate(fold_results):
                    warm_start[k] = w_fold
                    mse_path[idx_ratio, idx_alpha, k] = loss_fold

                mean_loss = np.mean(mse_path[idx_ratio, idx_alpha])
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    self.alpha_ = float(alpha)
                    self.l1_ratio_ = float(l1_ratio) if l1_ratio is not None else None

        # Refit on full dataset
        self.penalty.alpha = self.alpha_
        if hasattr(self.penalty, "l1_ratio"):
            self.penalty.l1_ratio = self.l1_ratio_
        super().fit(X, y)
        self.alphas_ = alphas
        self.mse_path_ = mse_path
        return self

    def predict(self, X):
        """Predict using the linear model."""
        X = np.asarray(X)
        if isinstance(self.datafit, (Logistic, QuadraticSVC)):
            return (X @ self.coef_ + self.intercept_ > 0).astype(int)
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        """Probability estimates for classification tasks."""
        if not isinstance(self.datafit, (Logistic, QuadraticSVC)):
            raise AttributeError(
                "predict_proba is only available for classification tasks"
            )
        X = np.asarray(X)
        decision = X @ self.coef_ + self.intercept_
        decision_2d = np.c_[-decision, decision]
        return softmax(decision_2d, copy=False)

    def score(self, X, y):
        """Return a 'higher = better' performance metric."""
        return -self._score(np.asarray(y), self.predict(X))
