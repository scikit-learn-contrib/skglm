"""
==============================
Show U-curve of regularization
==============================
Illustrate the sweet spot of regularization: not too much, not too little.
"""

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from libsvmdata import fetch_libsvm
from sklearn.model_selection import train_test_split

from skglm import Lasso


X, y = fetch_libsvm("rcv1.binary")
# we keep only 2000 features and samples
X = X[:, :2000]
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train, y_train = X_train[:2000], y_train[:2000]

alpha_max = norm(X_train.T @ y_train, ord=np.inf) / len(y_train)
alphas = np.geomspace(1, 1e-4) * alpha_max
mse = []
mse_test = []

clf = Lasso(fit_intercept=False, tol=1e-8, warm_start=True)
for idx, alpha in enumerate(alphas):
    clf.alpha = alpha
    clf.fit(X_train, y_train)
    mse.append(np.mean((y_train - clf.predict(X_train)) **2))
    mse_test.append(np.mean((y_test - clf.predict(X_test)) **2))

plt.close('all')
plt.semilogx(alphas, mse, label='train MSE')
plt.semilogx(alphas, mse_test, label='test MSE')
plt.legend()
plt.title("Mean squared error")
plt.xlabel(r"Lasso regularization strength $\lambda$")
plt.show(block=False)
