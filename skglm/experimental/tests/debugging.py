from skglm.experimental.quantile_huber import QuantileHuber
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.linear_model import QuantileRegressor

from skglm import GeneralizedLinearEstimator
from skglm.experimental.progressive_smoothing import ProgressiveSmoothingSolver
from skglm.experimental.pdcd_ws import PDCD_WS
from skglm.experimental.quantile_regression import Pinball
from skglm.penalties import L1
from skglm.solvers import FISTA


def pinball_loss(y_true, y_pred, tau=0.5):
    """Compute Pinball (quantile) loss."""
    residuals = y_true - y_pred
    return np.mean(np.where(residuals >= 0,
                            tau * residuals,
                            (1 - tau) * -residuals))


# Test 1: Check with more aggressive smoothing sequence
print("=== Test 1: More aggressive smoothing sequence ===")
np.random.seed(42)
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X = StandardScaler().fit_transform(X)

tau = 0.8
alpha = 0.1

# Reference solution
qr = QuantileRegressor(quantile=tau, alpha=alpha, fit_intercept=True,
                       solver="highs").fit(X, y)
y_pred_ref = qr.predict(X)
ref_loss = pinball_loss(y, y_pred_ref, tau=tau)

# More aggressive smoothing sequence
more_aggressive_seq = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]

pss = ProgressiveSmoothingSolver(
    smoothing_sequence=more_aggressive_seq,
    quantile=tau,
    alpha=alpha,
    verbose=True,
    smooth_solver=FISTA(max_iter=2000, tol=1e-8),
    nonsmooth_solver=PDCD_WS(max_iter=10000, max_epochs=5000, tol=1e-8)
).fit(X, y)

y_pred_pss = pss.predict(X)
pss_loss = pinball_loss(y, y_pred_pss, tau=tau)
rel_gap = (pss_loss - ref_loss) / ref_loss

print(f"Reference Pinball Loss: {ref_loss:.6f}")
print(f"PSS Pinball Loss: {pss_loss:.6f}")
print(f"Relative Gap: {rel_gap:.4f}")

# Check quantile distribution
residuals = y - y_pred_pss
n_pos = np.sum(residuals > 0)
n_neg = np.sum(residuals < 0)
actual_tau = n_pos / (n_pos + n_neg)
print(f"Expected quantile: {tau}")
print(f"Actual quantile: {actual_tau:.4f}")
print(f"Quantile error: {abs(actual_tau - tau):.4f}")

# Test 2: Diagnosing stage-by-stage progress
print("\n=== Test 2: Diagnosing stage-by-stage progress ===")
print("Stage results:")
for i, stage in enumerate(pss.stage_results_):
    print(
        f"Stage {i}: delta={stage['delta']:.4f}, obj_value={stage['obj_value']:.4f}, coef_norm={stage['coef_norm']:.4f}")

# Test 3: Compare with direct PDCD_WS
print("\n=== Test 3: Direct PDCD_WS for comparison ===")
pdcd_direct = PDCD_WS(max_iter=10000, max_epochs=5000, tol=1e-8, verbose=True)
estimator_direct = GeneralizedLinearEstimator(
    datafit=Pinball(tau),
    penalty=L1(alpha=alpha),
    solver=pdcd_direct,
).fit(X, y)

y_pred_direct = estimator_direct.predict(X)
pdcd_loss = pinball_loss(y, y_pred_direct, tau=tau)
print(f"Direct PDCD_WS Pinball Loss: {pdcd_loss:.6f}")
print(f"Direct PDCD_WS vs Reference Gap: {(pdcd_loss - ref_loss) / ref_loss:.4f}")

# Test 4: Verify QuantileHuber is working correctly
print("\n=== Test 4: Testing QuantileHuber directly ===")

# Test different deltas
deltas = [0.1, 0.01, 0.001]
for delta in deltas:
    qh = QuantileHuber(delta=delta, quantile=tau)
    huber_est = GeneralizedLinearEstimator(
        datafit=qh,
        penalty=L1(alpha=alpha),
        solver=FISTA(max_iter=2000, tol=1e-8),
    ).fit(X, y)

    y_pred_qh = huber_est.predict(X)
    qh_loss = pinball_loss(y, y_pred_qh, tau=tau)
    residuals_qh = y - y_pred_qh
    n_pos_qh = np.sum(residuals_qh > 0)
    n_neg_qh = np.sum(residuals_qh < 0)
    actual_tau_qh = n_pos_qh / (n_pos_qh + n_neg_qh)

    print(
        f"QuantileHuber(delta={delta}): Loss={qh_loss:.6f}, Actual tau={actual_tau_qh:.4f}")

# Test 5: Visualize the smoothing sequence effect
print("\n=== Test 5: Visualizing smoothing progression ===")

# Store predictions at each stage for visualization
stage_predictions = []
stage_losses = []
smoothed_solver = FISTA(max_iter=2000, tol=1e-8)

for delta in more_aggressive_seq:
    if tau == 0.5:
        datafit = Huber(delta=delta)
    else:
        datafit = QuantileHuber(delta=delta, quantile=tau)

    est = GeneralizedLinearEstimator(
        datafit=datafit,
        penalty=L1(alpha=alpha),
        solver=smoothed_solver,
    ).fit(X, y)

    y_pred_stage = est.predict(X)
    stage_predictions.append(y_pred_stage)
    stage_loss = pinball_loss(y, y_pred_stage, tau=tau)
    stage_losses.append(stage_loss)

    print(f"Delta={delta:.4f}: Loss={stage_loss:.6f}")

# Plot progression
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(stage_losses, 'o-', label='Smoothed stage losses')
plt.axhline(y=ref_loss, color='r', linestyle='--', label='Reference loss')
plt.xlabel('Stage')
plt.ylabel('Pinball Loss')
plt.legend()
plt.title(f'Progression of Pinball Loss (tau={tau})')
plt.grid(True)

plt.subplot(2, 1, 2)
for i, delta in enumerate(more_aggressive_seq):
    residuals_stage = y - stage_predictions[i]
    n_pos_stage = np.sum(residuals_stage > 0)
    n_neg_stage = np.sum(residuals_stage < 0)
    actual_tau_stage = n_pos_stage / (n_pos_stage + n_neg_stage)
    plt.plot(i, actual_tau_stage, 'o', label=f'delta={delta:.4f}')

plt.axhline(y=tau, color='r', linestyle='--', label='Target tau')
plt.xlabel('Stage')
plt.ylabel('Actual Quantile')
plt.legend()
plt.title('Progression of Actual Quantile')
plt.grid(True)

plt.tight_layout()
plt.savefig('progressive_smoothing_debug.png', dpi=150, bbox_inches='tight')
plt.close()

# Test 6: Debug the final convergence
print("\n=== Test 6: Analyzing final stage convergence ===")
print(f"Final coefficients norm: {np.linalg.norm(pss.coef_):.6f}")
print(f"Final intercept: {pss.intercept_:.6f}")

# Check gradients at the solution
residuals_final = y - pss.predict(X)
print(f"Mean residual: {np.mean(residuals_final):.6f}")
print(f"Residual std: {np.std(residuals_final):.6f}")

# Quantile check
quantile_actual = np.sum(residuals_final > 0) / len(residuals_final)
print(f"Final quantile check (should be {tau}): {quantile_actual:.4f}")
print(f"Quantile error: {abs(quantile_actual - tau):.4f}")

# Test 7: Look at coefficient differences
print("\n=== Test 7: Coefficient comparison ===")
coef_diff = np.linalg.norm(pss.coef_ - qr.coef_)
print(f"Coefficient difference (PSS vs reference): {coef_diff:.6f}")
print(f"Intercept difference: {abs(pss.intercept_ - qr.intercept_):.6f}")

# Test 8: Check if starting from a better initial point helps
print("\n=== Test 8: Testing with better initialization ===")
pss_better_init = ProgressiveSmoothingSolver(
    smoothing_sequence=[1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01],
    quantile=tau,
    alpha=alpha,
    verbose=False,
    smooth_solver=FISTA(max_iter=2000, tol=1e-8),
    nonsmooth_solver=PDCD_WS(max_iter=20000, max_epochs=10000, tol=1e-10)
)

# Initialize with reference solution
pss_better_init.fit(X, y)

y_pred_better = pss_better_init.predict(X)
better_loss = pinball_loss(y, y_pred_better, tau=tau)
better_gap = (better_loss - ref_loss) / ref_loss

print(f"Better initialized PSS Loss: {better_loss:.6f}")
print(f"Better initialized PSS Gap: {better_gap:.4f}")

residuals_better = y - y_pred_better
n_pos_better = np.sum(residuals_better > 0)
n_neg_better = np.sum(residuals_better < 0)
actual_tau_better = n_pos_better / (n_pos_better + n_neg_better)
print(f"Better initialized actual quantile: {actual_tau_better:.4f}")

# Alternative: Check if there's a fundamental issue with PDCD_WS for asymmetric quantiles
# Let's see what happens step-by-step in PDCD_WS iterations

# Add to the debugging script:
print("\n=== Test 9: Step-by-step PDCD_WS debugging ===")

# Start with the best smoothed solution
qh = QuantileHuber(delta=1e-4, quantile=tau)
huber_est = GeneralizedLinearEstimator(
    datafit=qh,
    penalty=L1(alpha=alpha),
    solver=FISTA(max_iter=5000, tol=1e-8),
).fit(X, y)

init_coef = huber_est.coef_.copy()
init_intercept = huber_est.intercept_

# Check initial quantile
y_pred_init = X @ init_coef
residuals_init = y - y_pred_init
n_pos_init = np.sum(residuals_init > 0)
n_neg_init = np.sum(residuals_init < 0)
actual_tau_init = n_pos_init / (n_pos_init + n_neg_init)
print(f"Initial quantile from smoothed solution: {actual_tau_init:.4f}")

# Now apply PDCD_WS with very verbose output
pdcd_detailed = PDCD_WS(max_iter=100, max_epochs=50, tol=1e-8, verbose=2)
est_detailed = GeneralizedLinearEstimator(
    datafit=Pinball(tau),
    penalty=L1(alpha=alpha),
    solver=pdcd_detailed,
).fit(X, y)

# Check final quantile
y_pred_final = est_detailed.predict(X)
residuals_final = y - y_pred_final
n_pos_final = np.sum(residuals_final > 0)
n_neg_final = np.sum(residuals_final < 0)
actual_tau_final = n_pos_final / (n_pos_final + n_neg_final)
print(f"Final quantile after PDCD_WS: {actual_tau_final:.4f}")

# Check convergence diagnostics
print(f"PDCD_WS converged: {hasattr(pdcd_detailed, 'converged_')}")
if hasattr(pdcd_detailed, 'converged_'):
    print(f"PDCD_WS convergence status: {pdcd_detailed.converged_}")
