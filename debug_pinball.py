from skglm.experimental import Pinball, PDCD_WS
from skglm import GeneralizedLinearEstimator
from skglm.penalties import L1
from skglm.solvers import ProxNewton, AndersonCD
from skglm.utils.data import make_correlated_data


solver = PDCD_WS(verbose=2)
# solver = AndersonCD(verbose=2, fit_intercept=False)

X, y, _ = make_correlated_data(random_state=0)
estimator = GeneralizedLinearEstimator(
    datafit=Pinball(.2),
    penalty=L1(alpha=1.),
    solver=solver,
)
estimator.fit(X, y)
