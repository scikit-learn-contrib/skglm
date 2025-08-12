import numpy as np
import pytest
from skglm.datafits import DoubleQuadratic, Quadratic


class TestDoubleQuadratic:
    
    def test_alpha_half_matches_quadratic(self):
        """Test that alpha=0.5 gives same results as Quadratic."""
        np.random.seed(42)
        X = np.random.randn(20, 5)
        y = np.random.randn(20)
        w = np.random.randn(5)
        Xw = X @ w
        
        quad = Quadratic()
        quad.initialize(X, y)
        
        dquad = DoubleQuadratic(alpha=0.5)
        dquad.initialize(X, y)
        
        # Test loss values
        assert np.allclose(quad.value(y, w, Xw), dquad.value(y, w, Xw))
        
        # Test gradients  
        assert np.allclose(quad.gradient(X, y, Xw), dquad.gradient(X, y, Xw))
    
    def test_asymmetric_behavior(self):
        """Test that asymmetric behavior works correctly."""
        # Simple test case with known residuals
        X = np.eye(4)
        y = np.zeros(4)
        w = np.array([1., -1., 2., -2.])  # residuals = [1, -1, 2, -2]
        Xw = X @ w
        
        dquad = DoubleQuadratic(alpha=0.3)
        dquad.initialize(X, y)
        
        loss = dquad.value(y, w, Xw)
        
        # Manual calculation with scaling: weights = 2 * [0.7, 0.3, 0.7, 0.3] = [1.4, 0.6, 1.4, 0.6]
        expected = (1/8) * (1.4*1 + 0.6*1 + 1.4*4 + 0.6*4)
        
        assert np.allclose(loss, expected)
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError):
            DoubleQuadratic(alpha=-0.1)
        
        with pytest.raises(ValueError):
            DoubleQuadratic(alpha=1.1)