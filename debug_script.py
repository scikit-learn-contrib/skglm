import timeit
import numpy as np

from skglm.utils import AndersonAcceleration


def test_anderson_acceleration(use_acc):
    n_features, max_iter = 2, 1000
    rho = np.array([0.5, 0.8])

    acc = AndersonAcceleration(K=5, n_features=n_features)
    w = np.ones(n_features)

    for _ in range(max_iter):
        if use_acc:
            acc.extrapolate(w)
        w = rho * w + 1

        if np.linalg.norm(w - 1 / (1 - rho), ord=np.inf) < 1e-10:
            break

    return w


print(timeit.timeit(lambda: test_anderson_acceleration(False), number=1000))
print(timeit.timeit(lambda: test_anderson_acceleration(True), number=1000))
