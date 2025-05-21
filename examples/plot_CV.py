"""
Cross-validation for Generalized Linear Models
============================================

This example demonstrates how to use cross-validation to select the optimal
regularization parameters for different types of generalized linear models.

We cover:
1. Lasso regression (L1 penalty)
2. Elastic Net regression (L1 + L2 penalty)
3. Logistic regression with L1 penalty
4. Logistic regression with Elastic Net penalty


Understanding Cross-Validation
----------------------------
Cross-validation (CV) is a technique to evaluate how well a model will perform
on unseen data. In this example, we use K-fold CV (K=5 by default) to:
1. Split the data into K folds
2. Train the model K times, each time using K-1 folds for training
3. Evaluate the model on the remaining fold
4. Average the results to get a robust estimate of model performance

The Process
----------
For each model type, we:
1. Generate synthetic data (or use real data)
2. Split it into training and test sets
3. Use CV to find the best regularization parameters
4. Train the final model with the best parameters
5. Evaluate on the test set

References
----------
[1] scikit-learn. (n.d.). Cross-validation: evaluating estimator performance.
    https://scikit-learn.org/stable/modules/cross_validation.html
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

from skglm.estimators import GeneralizedLinearEstimator
from skglm.datafits import Quadratic, Logistic
from skglm.penalties import L1, L1_plus_L2
from skglm.solvers import AndersonCD
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# 1. Lasso Regression Example
# --------------------------
print("1. Lasso Regression Example")
print("-" * 30)

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=20, noise=0.1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and fit Lasso with CV
lasso = GeneralizedLinearEstimator(
    datafit=Quadratic(),
    penalty=L1(alpha=1.0),
    solver=AndersonCD()
)
lasso.cross_validate(X_train, y_train, alphas='auto', cv=5,
                     scoring='neg_mean_squared_error')

# Evaluate on test set
y_pred = lasso.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Best alpha: {lasso.best_alpha_:.3f}")
print(f"Test MSE: {mse:.3f}")

# Plot CV scores
plt.figure(figsize=(10, 6))
plt.semilogx(lasso.alphas_, lasso.cv_scores_[None].mean(axis=0))
plt.axvline(lasso.best_alpha_, color='r', linestyle='--',
            label=f'Best alpha: {lasso.best_alpha_:.3f}')
plt.xlabel('Alpha')
plt.ylabel('CV Score')
plt.title('Lasso CV Scores')
plt.legend()
plt.grid(True)

# 2. Elastic Net Regression Example
# --------------------------------
print("\n2. Elastic Net Regression Example")
print("-" * 30)

# Create and fit Elastic Net with CV
enet = GeneralizedLinearEstimator(
    datafit=Quadratic(),
    penalty=L1_plus_L2(alpha=1.0, l1_ratio=0.5),
    solver=AndersonCD()
)
enet.cross_validate(X_train, y_train, alphas='auto',
                    l1_ratios=[0.1, 0.5, 0.9], cv=5, scoring='neg_mean_squared_error')

# Evaluate on test set
y_pred = enet.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Best alpha: {enet.best_alpha_:.3f}")
print(f"Best l1_ratio: {enet.best_l1_ratio_:.3f}")
print(f"Test MSE: {mse:.3f}")

# Plot CV scores for different l1_ratios
plt.figure(figsize=(10, 6))
for ratio in enet.cv_scores_:
    plt.semilogx(enet.alphas_, enet.cv_scores_[ratio].mean(axis=0),
                 label=f'l1_ratio={ratio}')
plt.axvline(enet.best_alpha_, color='r', linestyle='--',
            label=f'Best alpha: {enet.best_alpha_:.3f}')
plt.xlabel('Alpha')
plt.ylabel('CV Score')
plt.title('Elastic Net CV Scores')
plt.legend()
plt.grid(True)

# 3. Logistic Regression with L1 Penalty
# -------------------------------------
print("\n3. Logistic Regression with L1 Penalty")
print("-" * 30)

# Generate synthetic classification data
X, y = make_classification(n_samples=100, n_features=20, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and fit Logistic Regression with CV
logreg = GeneralizedLinearEstimator(
    datafit=Logistic(),
    penalty=L1(alpha=1.0),
    solver=AndersonCD()
)
logreg.cross_validate(X_train, y_train, alphas='auto', cv=5)

# Evaluate on test set
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Best alpha: {logreg.best_alpha_:.3f}")
print(f"Test Accuracy: {accuracy:.3f}")

# Plot CV scores
plt.figure(figsize=(10, 6))
plt.semilogx(logreg.alphas_, logreg.cv_scores_[None].mean(axis=0))
plt.axvline(logreg.best_alpha_, color='r', linestyle='--',
            label=f'Best alpha: {logreg.best_alpha_:.3f}')
plt.xlabel('Alpha')
plt.ylabel('CV Score')
plt.title('Logistic Regression CV Scores')
plt.legend()
plt.grid(True)

# 4. Logistic Regression with Elastic Net Penalty
# ---------------------------------------------
print("\n4. Logistic Regression with Elastic Net Penalty")
print("-" * 30)

# Create and fit Logistic Regression with Elastic Net penalty
logreg_enet = GeneralizedLinearEstimator(
    datafit=Logistic(),
    penalty=L1_plus_L2(alpha=1.0, l1_ratio=0.5),
    solver=AndersonCD()
)
logreg_enet.cross_validate(X_train, y_train, alphas='auto',
                           l1_ratios=[0.1, 0.5, 0.9], cv=5)

# Evaluate on test set
y_pred = logreg_enet.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Best alpha: {logreg_enet.best_alpha_:.3f}")
print(f"Best l1_ratio: {logreg_enet.best_l1_ratio_:.3f}")
print(f"Test Accuracy: {accuracy:.3f}")

# Plot CV scores for different l1_ratios
plt.figure(figsize=(10, 6))
for ratio in logreg_enet.cv_scores_:
    plt.semilogx(logreg_enet.alphas_, logreg_enet.cv_scores_[ratio].mean(axis=0),
                 label=f'l1_ratio={ratio}')
plt.axvline(logreg_enet.best_alpha_, color='r', linestyle='--',
            label=f'Best alpha: {logreg_enet.best_alpha_:.3f}')
plt.xlabel('Alpha')
plt.ylabel('CV Score')
plt.title('Logistic Regression with Elastic Net CV Scores')
plt.legend()
plt.grid(True)

plt.show()
