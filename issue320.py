import numpy as np
import time

from skglm import GeneralizedLinearEstimator
from skglm.datafits import Poisson
from skglm.penalties import L2
from skglm.solvers import LBFGS
from skglm.utils.data import make_correlated_data
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance, mean_absolute_error


def generate_correlated_poisson_data(
    n_samples=20000,
    n_features=50,
    rho=0.5,
    density=0.5,
    seed=42
):
    print("\n1. Generating synthetic correlated data for Poisson GLM...")
    print(
        f"   (n_samples={n_samples}, n_features={n_features}, "
        f"rho={rho}, density={density})"
    )

    # Use make_correlated_data to get X and w_true.
    X, _, w_true = make_correlated_data(
        n_samples=n_samples,
        n_features=n_features,
        rho=rho,
        snr=10,
        density=density,
        random_state=seed
    )

    # Define a true intercept
    intercept_true = -1.0

    # Calculate the linear predictor
    eta = intercept_true + X @ w_true

    # Apply the inverse link function
    eta = np.clip(eta, -15, 15)
    mu = np.exp(eta)

    # Generate the Poisson-distributed response variable
    rng = np.random.default_rng(seed)
    y = rng.poisson(mu)

    return X, y, w_true, intercept_true


def run_benchmark():
    """Main function to run the GLM benchmark."""

    # 1. Generate data
    # Parameters for data generation
    N_SAMPLES = 100000
    N_FEATURES = 1000
    RHO = 0.6
    DENSITY = 0.5  # Sparsity of true coefficients

    X, y_true, w_true, intercept_true = generate_correlated_poisson_data(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        rho=RHO,
        density=DENSITY,
        seed=42
    )

    # 2. Shared model parameters
    print("\n2. Setting up models...")
    alpha = 0.01  # L2 regularization strength
    tol = 1e-4  # Same tolerance as sklearn's PoissonRegressor
    iter_n = 1000  # Increase max_iter to allow convergence

    # 3a. Fit the GLM with skglm
    print("\n3a. Fitting the GLM with skglm...")
    estimator_skglm = GeneralizedLinearEstimator(
        datafit=Poisson(),
        # Using L2 penalty (Ridge) for LBFGS compatibility
        penalty=L2(alpha=alpha),
        solver=LBFGS(verbose=False, tol=tol, max_iter=iter_n, fit_intercept=True)
    )

    start_time_skglm = time.perf_counter()
    estimator_skglm.fit(X, y_true)
    end_time_skglm = time.perf_counter()
    skglm_fit_time = end_time_skglm - start_time_skglm
    print(f"   skglm fit complete in {skglm_fit_time:.4f} seconds.")

    # 3b. Fit the GLM with scikit-learn
    print("\n3b. Fitting the GLM with scikit-learn...")
    # PoissonRegressor in sklearn uses an L2 penalty.
    estimator_sklearn = PoissonRegressor(
        alpha=alpha,
        fit_intercept=True,
        tol=tol,
        solver='lbfgs',
        max_iter=iter_n
    )

    start_time_sklearn = time.time()
    estimator_sklearn.fit(X, y_true)
    end_time_sklearn = time.time()
    sklearn_fit_time = end_time_sklearn - start_time_sklearn
    print(f"   sklearn fit complete in {sklearn_fit_time:.4f} seconds.")

    # 4. Compare the results
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)

    # --- Coefficient Comparison ---
    print("\n--- Coefficient Comparison ---")

    # Intercept
    print(f"{'Parameter':<20} | {'Ground Truth':<15} | "
          f"{'skglm Fit':<15} | {'sklearn Fit':<15}")
    print("-" * 75)
    print(f"{'Intercept':<20} | {intercept_true:<15.4f} | "
          f"{estimator_skglm.intercept_:<15.4f} | "
          f"{estimator_sklearn.intercept_:<15.4f}")

    # MAE of Coefficients
    mae_skglm = mean_absolute_error(w_true, estimator_skglm.coef_)
    mae_sklearn = mean_absolute_error(w_true, estimator_sklearn.coef_)
    print(f"\n{'MAE vs. w_true':<20} | {'':<15} | "
          f"{mae_skglm:<15.6f} | {mae_sklearn:<15.6f}")

    # Spot-check of first 5 coefficients
    print("\nSpot-check of first 5 coefficients:")
    print(f"{'Parameter':<12} | {'Ground Truth':<15} | "
          f"{'skglm Fit':<15} | {'sklearn Fit':<15}")
    print("-" * 65)
    for i in range(min(5, N_FEATURES)):
        print(
            f"w_{i:<10} | {w_true[i]:<15.4f} | "
            f"{estimator_skglm.coef_[i]:<15.4f} | "
            f"{estimator_sklearn.coef_[i]:<15.4f}")

    # --- Timing Comparison ---
    print("\n--- Fitting Time Comparison ---")
    print(f"skglm (LBFGS):   {skglm_fit_time:.4f} seconds")
    print(f"sklearn (L-BFGS):     {sklearn_fit_time:.4f} seconds")
    if skglm_fit_time < sklearn_fit_time:
        speedup = sklearn_fit_time / \
            skglm_fit_time if skglm_fit_time > 0 else float('inf')
        print(f" >> skglm was {speedup:.2f}x faster.")
    else:
        speedup = skglm_fit_time / \
            sklearn_fit_time if sklearn_fit_time > 0 else float('inf')
        print(f" >> sklearn was {speedup:.2f}x faster.")

    # --- Performance Metrics Comparison ---
    def calculate_metrics(estimator, X, y_true):
        y_pred = estimator.predict(X)
        # clip to avoid log(0) in deviance calculation
        y_pred = np.clip(y_pred, 1e-9, None)
        dev_model = len(y_true) * mean_poisson_deviance(y_true, y_pred)
        return dev_model

    dev_model_skglm = calculate_metrics(estimator_skglm, X, y_true)
    dev_model_sklearn = calculate_metrics(estimator_sklearn, X, y_true)

    # Null deviance
    y_null = np.full_like(y_true, fill_value=y_true.mean(), dtype=float)
    dev_null = len(y_true) * mean_poisson_deviance(y_true, y_null)

    pseudo_r2_skglm = 1.0 - (dev_model_skglm / dev_null)
    pseudo_r2_sklearn = 1.0 - (dev_model_sklearn / dev_null)

    print("\n--- Performance Metrics ---")
    print(f"{'Metric':<30} | {'skglm':<15} | {'sklearn':<15}")
    print("-" * 65)
    print(f"{'Model Deviance':<30} | {dev_model_skglm:<15,.2f} | "
          f"{dev_model_sklearn:<15,.2f}")
    print(f"{'Null Deviance':<30} | {dev_null:<15,.2f} | {dev_null:<15,.2f}")
    print(f"{'Pseudo R² (Deviance Explained)':<30} | "
          f"{pseudo_r2_skglm:<15.4f} | {pseudo_r2_sklearn:<15.4f}")


if __name__ == "__main__":
    run_benchmark()
