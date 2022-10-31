import numpy as np
from numpy.linalg import norm


def fercoq_bianchi(X, y, datafit, penalty, max_iter=100, tol=1e-4,
                   verbose=0, random_state=0):
    n_samples, n_features = X.shape
    rng = np.random.RandomState(random_state)
    all_features = np.arange(n_features)

    stop_crit = 0.
    p_objs_out = []

    # primal variable
    w = np.zeros(n_features)
    Xw = np.zeros(n_samples)

    # dual variable
    z = np.zeros(n_samples)
    z_bar = np.zeros(n_samples)

    # constants verifies: tau_j < 1 / (||X|| * (2n-1) * sigma * ||X_j||^2)
    sigma = 1 / ((2 * n_features - 1) * norm(X, ord=2))
    tau = 1 / (X ** 2).sum(axis=0)  # 1 / squared norm of columns

    for iter in range(max_iter):

        # random CD
        for j in rng.choice(n_features, n_features):
            # dual update
            z_bar = datafit.prox_conjugate(y, X, z + sigma * Xw, sigma)
            z += (z_bar - z) / n_features

            # primal update
            old_w_j = w[j]
            w[j] = penalty.prox_1d(old_w_j - tau[j] * (X[:, j] @ (2 * z_bar - z)),
                                   tau[j], j)

            # keep Xw synchronized with X @ W
            delta_w_j = w[j] - old_w_j
            if delta_w_j != 0:
                Xw += delta_w_j * X[:, j]

        # check convergence
        if iter % 10 == 0:
            otp = penalty.subdiff_distance(w, X.T @ z, all_features)
            stop_crit = np.max(otp)

            if verbose:
                p_obj = datafit.value(y, w, Xw) + penalty.value(w)
                print(
                    f"Iteration {iter+1}: {p_obj:.10f}, "
                    f"stopping crit: {stop_crit:.2e}"
                )

            if stop_crit <= tol:
                break

        p_obj = datafit.value(y, w, Xw) + penalty.value(w)
        p_objs_out.append(p_obj)
    return w, np.asarray(p_objs_out), stop_crit


if __name__ == '__main__':
    pass
