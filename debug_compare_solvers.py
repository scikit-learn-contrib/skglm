import time
import numpy as np
from skglm.covariance import GraphicalLasso
from skglm.utils.data import make_dummy_covariance_data

S, X, Theta_true, lmbd_max = make_dummy_covariance_data(200, 50)


print("Comparing GraphicalLasso solvers")
print("=" * 40)

for solver in ['barebones', 'epoch', 'standard_gramcd']:
    glasso = GraphicalLasso(alpha=0.1, solver=solver, max_iter=100, tol=1e-4)
    glasso.fit(X.copy(), mode='empirical')

results = {}

for solver in ['barebones', 'epoch', 'standard_gramcd']:
    X_copy = X.copy()

    times = []
    for _ in range(5):
        glasso = GraphicalLasso(alpha=0.1, solver=solver,
                                max_iter=100, tol=1e-4)
        start_time = time.time()
        glasso.fit(X_copy, mode='empirical')
        times.append(time.time() - start_time)

    # Check for NaN/Inf in the precision matrix
    precision = glasso.precision_.copy()
    if np.any(np.isnan(precision)):
        print(f"Warning: {solver} precision_ contains NaN values!")
    if np.any(np.isinf(precision)):
        print(f"Warning: {solver} precision_ contains Inf values!")

    avg_time = np.mean(times)
    std_time = np.std(times)

    results[solver] = {
        'time': avg_time,
        'std': std_time,
        'n_iter': glasso.n_iter_,
        'precision': glasso.precision_.copy()
    }

    print(f"{solver:15s}: {avg_time:.4f}s (±{std_time:.4f}s), {glasso.n_iter_} iter")

print("\nPrecision differences:")
diff_barebones_epoch = np.linalg.norm(
    results['barebones']['precision'] - results['epoch']['precision']
)
print(f"Barebones vs Epoch: {diff_barebones_epoch:.2e}")
diff_epoch_standard = np.linalg.norm(
    results['epoch']['precision'] - results['standard_gramcd']['precision']
)
print(f"Epoch vs Standard: {diff_epoch_standard:.2e}")

print("\nSpeedup ratios:")
for solver1 in ['barebones', 'epoch']:
    for solver2 in ['epoch', 'standard_gramcd']:
        if solver1 != solver2:
            speedup = results[solver2]['time'] / results[solver1]['time']
            print(f"{solver1} vs {solver2}: {speedup:.2f}x")
