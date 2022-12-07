"""
=========================================
Sparse recovery with non-convex penalties
=========================================
Illustrate the superior performance of penalties for sparse recovery.
"""

# Author: Mathurin Massias
#         Quentin Bertrand
#         Quentin Klopfenstein

import numpy as np
from scipy import stats
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error

from skglm.utils.data import make_correlated_data
from skglm.solvers import AndersonCD, FISTA
from skglm.datafits import Quadratic
from skglm.utils.jit_compilation import compiled_clone
from skglm.penalties import L1, MCPenalty, L0_5, L2_3, SCAD, SLOPE

cmap = plt.get_cmap('tab10')

# %%
# First we simulate noisy linear regression data with sparse true coefficients
# ``w_true``
n_features = 1000
density = 0.1
np.random.seed(0)
supp = np.random.choice(n_features, size=int(density * n_features),
                        replace=False)
w_true = np.zeros(n_features)
w_true[supp] = 1
X_, y_, w_true = make_correlated_data(
    n_samples=1000, n_features=n_features, snr=5, random_state=2,
    rho=0.5, w_true=w_true)

# %%
# In order to circumvent issues with MCP, the data is standardized.
X_ /= norm(X_, axis=0) / np.sqrt(len(X_))
X, X_test, y, y_test = train_test_split(X_, y_, test_size=0.5)


# %%
# To tune the regularization strengths of all estimators, we parametrize them as
# fraction of :math:`\lambda_{max}`, the minimal L1 regularization strength giving a null
# solution for the Lasso.
alpha_max = norm(X.T @ y, ord=np.inf) / len(y)

n_alphas = 30
alphas = alpha_max * np.geomspace(1, 1e-2, num=n_alphas)

# Benjamini-Hochberg sequence for SLOPE:
slope_seq = stats.norm(loc=0, scale=1).ppf(
    1 - np.arange(1, n_features+1) * 0.1 / (2 * n_features))


# %%
# Instanciate models with the skglm API, through ``datafit`` and ``penalty``:
datafit = compiled_clone(Quadratic())
datafit.initialize(X, y)

penalties = {}
penalties['lasso'] = L1(alpha=1)
# penalties['mcp'] = MCPenalty(alpha=1, gamma=3)
# penalties['scad'] = SCAD(alpha=1, gamma=3)
# penalties['l05'] = L0_5(alpha=1)
# penalties['l23'] = L2_3(alpha=1)
penalties['slope'] = SLOPE(alphas=1)

colors = {}
colors['lasso'] = cmap(0)
colors['mcp'] = cmap(1)
colors['scad'] = cmap(2)
colors['l05'] = cmap(3)
colors['l23'] = cmap(4)
colors['slope'] = cmap(5)

f1 = {}
estimation_error = {}
prediction_error = {}
l0 = {}
mse_ref = mean_squared_error(np.zeros_like(y_test), y_test)

solver = AndersonCD(ws_strategy="fixpoint", fit_intercept=False)
fista = FISTA(tol=1e-7, opt_strategy="fixpoint")  # important for SLOPE

for idx, penalty_name in enumerate(penalties.keys()):
    print(f'Running {penalty_name}...')
    penalty = penalties[penalty_name]
    if penalty_name == "slope":
        coefs_path = np.zeros([n_features, n_alphas])
        w = np.zeros(n_features)
        Xw = X @ w
        for alpha_idx, alpha in enumerate(alphas):
            penalty.alphas = alpha * slope_seq
            w = fista.solve(X, y, datafit, compiled_clone(penalty), w, Xw)[0]
            Xw = X @ w  # warm start
            coefs_path[:, alpha_idx] = w
    else:
        coefs_path = solver.path(
            X, y, datafit, compiled_clone(penalty),
            alphas=alphas)[1]

    f1_temp = np.zeros(n_alphas)
    prediction_error_temp = np.zeros(n_alphas)

    for j, w in enumerate(coefs_path.T):
        f1_temp[j] = f1_score(w != 0, w_true != 0)
        prediction_error_temp[j] = mean_squared_error(X_test @ w, y_test) / mse_ref

    f1[penalty_name] = f1_temp
    prediction_error[penalty_name] = prediction_error_temp


# %%
# Plot recovery results per penalty:
full_names = {'lasso': "Lasso"}
full_names['mcp'] = r"MCP, $\gamma=%s$" % 3
full_names['scad'] = r"SCAD, $\gamma=%s$" % 3
full_names['l05'] = r"$\ell_{1/2}$"
full_names['l23'] = r"$\ell_{2/3}$"
full_names['slope'] = "SLOPE"


plt.close('all')
fig, axarr = plt.subplots(2, 1, sharex=True, sharey=False, figsize=[
                          6.3, 4], constrained_layout=True)

for idx, penalty_name in enumerate(penalties.keys()):

    axarr[0].semilogx(
        alphas / alphas[0], f1[penalty_name], label=full_names[penalty_name],
        c=colors[penalty_name])

    axarr[1].semilogx(
        alphas / alphas[0], prediction_error[penalty_name],
        label=full_names[penalty_name], c=colors[penalty_name])

    max_f1 = np.argmax(f1[penalty_name])
    axarr[0].vlines(
        x=alphas[max_f1] / alphas[0], ymin=0,
        ymax=np.max(f1[penalty_name]),
        color=colors[penalty_name], linestyle='--')
    line1 = axarr[0].plot(
        [alphas[max_f1] / alphas[0]], 0, clip_on=False,
        marker='X', color=colors[penalty_name], markersize=12)

    min_error = np.argmin(prediction_error[penalty_name])

    lims = axarr[1].get_ylim()
    axarr[1].vlines(
        x=alphas[min_error] / alphas[0], ymin=0,
        ymax=np.min(prediction_error[penalty_name]),
        color=colors[penalty_name], linestyle='--')

    line2 = axarr[1].plot(
        [alphas[min_error] / alphas[0]], 0, clip_on=False,
        marker='X', color=colors[penalty_name], markersize=12)
    axarr[1].set_xlabel(r"$\lambda / \lambda_{\mathrm{max}}$")
    axarr[0].set_ylabel("F1-score")
    axarr[0].set_ylim(ymin=0, ymax=1.0)
    axarr[1].set_ylim(ymin=0, ymax=lims[1])
    axarr[1].set_ylabel("pred. RMSE left-out")
    axarr[0].legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
        mode="expand", borderaxespad=0, ncol=5)

plt.show(block=False)
