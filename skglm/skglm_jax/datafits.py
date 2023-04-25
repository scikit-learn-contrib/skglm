from jax.numpy.linalg import norm as jnorm


class QuadraticJax:

    def value(self, X, y, w):
        n_samples = X.shape[0]
        return ((X @ w - y) ** 2).sum() / (2. * n_samples)

    def gradient_1d(self, X, y, w, j):
        n_samples = X.shape[0]
        return X[:, j] @ (X @ w - y) / n_samples

    def get_features_lipschitz_cst(self, X, y):
        n_samples = X.shape[0]
        return jnorm(X, ord=2, axis=0) ** 2 / n_samples
