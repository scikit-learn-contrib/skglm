from random import random
import numpy as np
from benchopt.datasets.simulated import make_correlated_data

from skglm import LinearSVC, Lasso

X, y, _ = make_correlated_data(50, 100, random_state=0)

y = np.sign(y)

clf = LinearSVC(verbose=2, tol=1e-10, warm_start=True)
clf.fit(X, y)

clf.fit(X, y)


reg = Lasso(alpha=0.01, verbose=2, warm_start=True)
print('##')
reg.fit(X, y)
print('##')
reg.fit(X, y)
