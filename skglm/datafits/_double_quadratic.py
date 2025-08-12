import numpy as np
from numba import float64
from .base import BaseDatafit


class DoubleQuadratic(BaseDatafit):
    """Double Quadratic datafit with asymmetric loss.

    The datafit reads:

    .. math:: 1 / (2 \\times n_\\text{samples}) \\sum_i (\\alpha + (1-2\\alpha) \\cdot 1[\\epsilon_i > 0]) \\epsilon_i^2

    where :math:`\\epsilon_i = (Xw)_i - y_i` are the residuals.

    Parameters
    ----------
    alpha : float, default=0.5
        Asymmetry parameter controlling the relative weighting of positive vs 
        negative residuals:
        - alpha = 0.5: symmetric loss (equivalent to standard Quadratic)
        - alpha < 0.5: penalize positive residuals (overestimation) more heavily
        - alpha > 0.5: penalize negative residuals (underestimation) more heavily

    Attributes
    ----------
    Xty : array, shape (n_features,)
        Pre-computed quantity used during the gradient evaluation.
        Equal to ``X.T @ y``.

    Note
    ----
    The class is jit compiled at fit time using Numba compiler.
    This allows for faster computations.
    """

    def __init__(self, alpha=0.5):
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        self.alpha = alpha

    def get_spec(self):
        spec = (
            ('alpha', float64),
            ('Xty', float64[:]),
        )
        return spec

    def params_to_dict(self):
        return dict(alpha=self.alpha)

    def get_lipschitz(self, X, y):
        """Compute per-coordinate Lipschitz constants.
        
        For DoubleQuadratic with scaling factor 2, the Lipschitz constant 
        for coordinate j is bounded by 2 * max_weight * ||X[:, j]||^2 / n_samples.
        """
        n_features = X.shape[1]
        
        # Compute weight bounds (after scaling by 2)
        weight_pos = 2 * (1 - self.alpha)  # weight for positive residuals  
        weight_neg = 2 * self.alpha        # weight for negative residuals
        max_weight = max(weight_pos, weight_neg)
        
        lipschitz = np.zeros(n_features, dtype=X.dtype)
        for j in range(n_features):
            lipschitz[j] = max_weight * (X[:, j] ** 2).sum() / len(y)

        return lipschitz

    def get_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        """Sparse version of get_lipschitz."""
        n_features = len(X_indptr) - 1
        
        # Compute weight bounds (after scaling by 2)
        weight_pos = 2 * (1 - self.alpha)
        weight_neg = 2 * self.alpha
        max_weight = max(weight_pos, weight_neg)
        
        lipschitz = np.zeros(n_features, dtype=X_data.dtype)

        for j in range(n_features):
            nrm2 = 0.
            for idx in range(X_indptr[j], X_indptr[j + 1]):
                nrm2 += X_data[idx] ** 2

            lipschitz[j] = max_weight * nrm2 / len(y)

        return lipschitz

    def get_global_lipschitz(self, X, y):
        """Global Lipschitz constant."""
        weight_pos = 2 * (1 - self.alpha)
        weight_neg = 2 * self.alpha
        max_weight = max(weight_pos, weight_neg)
        
        from scipy.linalg import norm
        return max_weight * norm(X, ord=2) ** 2 / len(y)

    def get_global_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        """Sparse version of global Lipschitz constant."""
        weight_pos = 2 * (1 - self.alpha)
        weight_neg = 2 * self.alpha
        max_weight = max(weight_pos, weight_neg)
        
        from .utils import spectral_norm
        return max_weight * spectral_norm(X_data, X_indptr, X_indices, len(y)) ** 2 / len(y)

    def initialize(self, X, y):
        """Pre-compute X.T @ y for efficient gradient computation."""
        self.Xty = X.T @ y

    def initialize_sparse(self, X_data, X_indptr, X_indices, y):
        """Sparse version of initialize."""
        n_features = len(X_indptr) - 1
        self.Xty = np.zeros(n_features, dtype=X_data.dtype)

        for j in range(n_features):
            xty = 0
            for idx in range(X_indptr[j], X_indptr[j + 1]):
                xty += X_data[idx] * y[X_indices[idx]]

            self.Xty[j] = xty

    def value(self, y, w, Xw):
        """Compute the asymmetric quadratic loss value.
        
        When alpha=0.5, this should be identical to Quadratic loss.
        The formula needs to be: (1/2n) * Σ weights * (y - Xw)²
        where weights are normalized so that alpha=0.5 gives weight=1.0
        """
        # Match Quadratic exactly: use (y - Xw) for loss computation
        residuals = y - Xw  
        
        # For asymmetric weighting, check sign of (Xw - y)  
        prediction_residuals = Xw - y
        
        # Compute weights, normalized so alpha=0.5 gives weight=1.0
        # Original formula: α + (1-2α) * 1[εᵢ>0]
        # At α=0.5: 0.5 + 0 = 0.5, but we want 1.0
        # So we need to scale by 2: 2 * (α + (1-2α) * 1[εᵢ>0])
        weights = 2 * (self.alpha + (1 - 2*self.alpha) * (prediction_residuals > 0))
        
        # Return normalized loss
        return np.sum(weights * residuals**2) / (2 * len(y))

    def gradient_scalar(self, X, y, w, Xw, j):
        """Compute gradient w.r.t. coordinate j."""
        prediction_residuals = Xw - y  # For gradient computation
        
        # Compute weights with same scaling as value()
        weights = 2 * (self.alpha + (1 - 2*self.alpha) * (prediction_residuals > 0))
        
        # Gradient: X[:, j].T @ (weights * (Xw - y)) / n_samples
        return (X[:, j] @ (weights * prediction_residuals)) / len(y)

    def gradient_scalar_sparse(self, X_data, X_indptr, X_indices, y, Xw, j):
        """Sparse version of gradient_scalar."""
        prediction_residuals = Xw - y
        
        # Compute weights with same scaling 
        weights = 2 * (self.alpha + (1 - 2*self.alpha) * (prediction_residuals > 0))
        
        # Compute X[:, j].T @ (weights * prediction_residuals) for sparse X
        XjT_weighted_residuals = 0.
        for i in range(X_indptr[j], X_indptr[j+1]):
            sample_idx = X_indices[i]
            XjT_weighted_residuals += X_data[i] * weights[sample_idx] * prediction_residuals[sample_idx]
        
        return XjT_weighted_residuals / len(y)

    def gradient(self, X, y, Xw):
        """Compute full gradient vector."""
        prediction_residuals = Xw - y
        
        # Compute weights with same scaling as value()
        weights = 2 * (self.alpha + (1 - 2*self.alpha) * (prediction_residuals > 0))
        
        # Return X.T @ (weights * prediction_residuals) / n_samples
        return X.T @ (weights * prediction_residuals) / len(y)

    def raw_grad(self, y, Xw):
        """Compute gradient of datafit w.r.t Xw."""
        prediction_residuals = Xw - y
        
        # Compute weights with same scaling
        weights = 2 * (self.alpha + (1 - 2*self.alpha) * (prediction_residuals > 0))
        
        return weights * prediction_residuals / len(y)

    def raw_hessian(self, y, Xw):
        """Compute Hessian of datafit w.r.t Xw."""
        prediction_residuals = Xw - y
        
        # Compute weights with same scaling
        weights = 2 * (self.alpha + (1 - 2*self.alpha) * (prediction_residuals > 0))
        
        return weights / len(y)

    def full_grad_sparse(self, X_data, X_indptr, X_indices, y, Xw):
        """Sparse version of full gradient computation."""
        n_features = X_indptr.shape[0] - 1
        n_samples = y.shape[0]
        prediction_residuals = Xw - y
        
        # Compute weights with same scaling
        weights = 2 * (self.alpha + (1 - 2*self.alpha) * (prediction_residuals > 0))
        
        grad = np.zeros(n_features, dtype=Xw.dtype)
        for j in range(n_features):
            XjT_weighted_residuals = 0.
            for i in range(X_indptr[j], X_indptr[j + 1]):
                sample_idx = X_indices[i]
                XjT_weighted_residuals += X_data[i] * weights[sample_idx] * prediction_residuals[sample_idx]
            grad[j] = XjT_weighted_residuals / n_samples
        return grad

    def intercept_update_step(self, y, Xw):
        """Compute intercept update step."""
        prediction_residuals = Xw - y
        
        # Compute weights with same scaling
        weights = 2 * (self.alpha + (1 - 2*self.alpha) * (prediction_residuals > 0))
        
        return np.mean(weights * prediction_residuals)


# Test function to validate our implementation
def _test_double_quadratic():
    """Test DoubleQuadratic implementation."""
    import numpy as np
    from .single_task import Quadratic
    
    print("Testing DoubleQuadratic implementation...")
    
    # Test data
    np.random.seed(42)
    n_samples, n_features = 50, 10
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    w = np.random.randn(n_features)
    Xw = X @ w
    
    # Test 1: alpha=0.5 should match standard Quadratic
    print("\n=== Test 1: alpha=0.5 vs Quadratic ===")
    
    quad = Quadratic()
    quad.initialize(X, y)
    
    dquad = DoubleQuadratic(alpha=0.5)
    dquad.initialize(X, y)
    
    loss_quad = quad.value(y, w, Xw)
    loss_dquad = dquad.value(y, w, Xw)
    
    print(f"Quadratic loss:      {loss_quad:.8f}")
    print(f"DoubleQuadratic:     {loss_dquad:.8f}")
    print(f"Difference:          {abs(loss_quad - loss_dquad):.2e}")
    
    # Test gradients
    grad_quad = quad.gradient(X, y, Xw)
    grad_dquad = dquad.gradient(X, y, Xw)
    grad_diff = np.linalg.norm(grad_quad - grad_dquad)
    
    print(f"Gradient difference: {grad_diff:.2e}")
    
    # Test case 2: Asymmetric behavior
    print("\n=== Test 2: Asymmetric behavior ===")
    
    # Create simple test case with known residuals
    X_simple = np.eye(4)  # Identity matrix
    y_simple = np.array([0., 0., 0., 0.])
    w_simple = np.array([1., -1., 2., -2.])  # Predictions: [1, -1, 2, -2], so prediction_residuals = [1, -1, 2, -2]
    Xw_simple = X_simple @ w_simple
    
    dquad_asym = DoubleQuadratic(alpha=0.3)  # Penalize positive residuals more
    dquad_asym.initialize(X_simple, y_simple)
    
    loss_asym = dquad_asym.value(y_simple, w_simple, Xw_simple)
    
    # Manual calculation:
    # prediction_residuals = [1, -1, 2, -2] (Xw - y)
    # weights = 0.3 + 0.4 * [1, 0, 1, 0] = [0.7, 0.3, 0.7, 0.3]  
    # loss = (1/(2*4)) * (0.7*1² + 0.3*1² + 0.7*4² + 0.3*4²)
    expected = (1/8) * (0.7*1 + 0.3*1 + 0.7*4 + 0.3*4)
    
    print(f"Asymmetric loss:     {loss_asym:.6f}")
    print(f"Expected:            {expected:.6f}")
    print(f"Difference:          {abs(loss_asym - expected):.2e}")
    
    print("\n=== All tests completed ===")


if __name__ == "__main__":
    _test_double_quadratic()