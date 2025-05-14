import numpy as np
from numpy.linalg import norm
from numba import float64
from skglm.datafits.base import BaseDatafit
from skglm.utils.sparse_ops import spectral_norm


class QuantileHuber(BaseDatafit):
    r"""Smoothed approximation of the pinball loss for quantile regression.

    This class implements a smoothed version of the pinball loss used in quantile
    regression. The original non-smooth pinball loss is defined as:

    .. math::

        \rho_\tau(r) =
        \begin{cases}
        \tau r & \text{if } r \geq 0 \\
        (\tau - 1) r & \text{if } r < 0
        \end{cases}

    where :math:`r = y - X\beta` is the residual and :math:`\tau \in (0, 1)` is
    the desired quantile level.

    The smoothed version (Huberized pinball loss) replaces the non-differentiable
    point at r=0 with a quadratic region for |r| < δ:

    .. math::

        \rho_\tau^\delta(r) =
        \begin{cases}
        \tau r - \frac{\delta}{2} & \text{if } r \geq \delta \\
        \frac{\tau r^2}{2\delta} & \text{if } 0 \leq r < \delta \\
        \frac{(1-\tau) r^2}{2\delta} & \text{if } -\delta < r < 0 \\
        (\tau - 1) r - \frac{\delta}{2} & \text{if } r \leq -\delta
        \end{cases}

    This can be more compactly written as:

    .. math::
        \rho_\tau^\delta(r) = \begin{cases}
            \tau r - \frac{\delta}{2} & \text{if } r \geq \delta \\
            \frac{r^2}{2\delta} \cdot s(r) & \text{if } |r| < \delta \\
            (\tau - 1) r - \frac{\delta}{2} & \text{if } r \leq -\delta
        \end{cases}

    where s(r) is τ when r ≥ 0 and (1-τ) when r < 0.

    Parameters
    ----------
    delta : float
        Smoothing parameter. Controls the width of the quadratic region.
        Smaller values make the approximation closer to the original
        non-smooth pinball loss, but may lead to numerical instability.

    quantile : float, default=0.5
        Desired quantile level between 0 and 1. When 0.5, the loss is
        symmetric (equivalent to Huber loss). For other values, the loss
        is asymmetric.

    Attributes
    ----------
    delta : float
        Current smoothing parameter.

    quantile : float
        Current quantile level.

    Notes
    -----
    The gradient of the smoothed loss is continuous and defined as:

    .. math::

        \nabla \rho_\tau^\delta(r) =
        \begin{cases}
        \tau & \text{if } r \geq \delta \\
        \frac{\tau r}{\delta} & \text{if } 0 \leq r < \delta \\
        \frac{(1-\tau) r}{\delta} & \text{if } -\delta < r < 0 \\
        \tau - 1 & \text{if } r \leq -\delta
        \end{cases}

    This can be more compactly written as:

    .. math::
        \nabla \rho_\tau^\delta(r) = \begin{cases}
            \tau & \text{if } r \geq \delta \\
            \frac{r}{\delta} \cdot s(r) & \text{if } |r| < \delta \\
            \tau - 1 & \text{if } r \leq -\delta
        \end{cases}

    where s(r) is τ when r ≥ 0 and (1-τ) when r < 0.

    As δ approaches 0, the smoothed loss converges to the original non-smooth
    pinball loss, which is exactly the quantile regression objective.

    References
    ----------
    Chen, C. (2007). A Finite Smoothing Algorithm for Quantile Regression.
    Journal of Computational and Graphical Statistics, 16(1), 136–164.
    http://www.jstor.org/stable/27594233

    Examples
    --------
    >>> from skglm.experimental.quantile_huber import QuantileHuber
    >>> import numpy as np
    >>> # Create a loss with smoothing parameter 0.1 for the 80th percentile
    >>> loss = QuantileHuber(delta=0.1, quantile=0.8)
    >>>
    >>> # Compute loss values for different residuals
    >>> residuals = np.array([-1.0, -0.05, 0.0, 0.05, 1.0])
    >>> for r in residuals:
    ...     loss_val, grad_val = loss._loss_and_grad_scalar(r)
    ...     print(f"Residual: {r:.2f}, Loss: {loss_val:.4f}, Gradient: {grad_val:.4f}")
    """

    def __init__(self, delta, quantile=0.5):
        if not 0 < quantile < 1:
            raise ValueError("quantile must be between 0 and 1")
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.delta = float(delta)
        self.quantile = float(quantile)

    def get_spec(self):
        """Get numba specification for JIT compilation."""
        spec = (
            ('delta', float64),
            ('quantile', float64),
        )
        return spec

    def params_to_dict(self):
        """Return parameters as a dictionary."""
        return dict(delta=self.delta, quantile=self.quantile)

    def get_lipschitz(self, X, y):
        """
        Compute coordinate-wise Lipschitz constants for the gradient.

        For the smoothed pinball loss, the Lipschitz constant is proportional
        to 1/delta, making it more challenging to optimize as delta gets smaller.
        """
        n_samples = len(y)
        # The max(τ, 1-τ) factor accounts for the asymmetry of the loss
        weight = max(self.quantile, 1 - self.quantile)

        # For each feature, compute L_j = weight * ||X_j||^2 / (n * delta)
        lipschitz = weight * (X ** 2).sum(axis=0) / (n_samples * self.delta)
        return lipschitz

    def get_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        """Compute coordinate-wise Lipschitz constants for sparse X."""
        n_samples = len(y)
        n_features = len(X_indptr) - 1
        weight = max(self.quantile, 1 - self.quantile)

        lipschitz = np.zeros(n_features, dtype=X_data.dtype)
        for j in range(n_features):
            nrm2 = 0.0
            for idx in range(X_indptr[j], X_indptr[j + 1]):
                nrm2 += X_data[idx] ** 2
            lipschitz[j] = weight * nrm2 / (n_samples * self.delta)
        return lipschitz

    def get_global_lipschitz(self, X, y):
        """
        Compute the global Lipschitz constant for the gradient.

        Uses the spectral norm of X to find a global constant.
        """
        n_samples = len(y)
        weight = max(self.quantile, 1 - self.quantile)
        return weight * norm(X, 2) ** 2 / (n_samples * self.delta)

    def get_global_lipschitz_sparse(self, X_data, X_indptr, X_indices, y):
        """Compute the global Lipschitz constant for sparse X."""
        n_samples = len(y)
        weight = max(self.quantile, 1 - self.quantile)
        return (
            weight
            * spectral_norm(X_data, X_indptr, X_indices, n_samples) ** 2
            / (n_samples * self.delta)
        )

    def _loss_and_grad_scalar(self, residual):
        """
        Calculate the smoothed pinball loss and its gradient for a single residual.

        This implements the core mathematical formulation of the quantile Huber loss.

        Parameters
        ----------
        residual : float
            The residual value r = y - Xβ

        Returns
        -------
        loss : float
            The value of the smoothed pinball loss at this residual

        gradient : float
            The gradient of the smoothed pinball loss at this residual
        """
        tau = self.quantile
        delta = self.delta
        abs_r = abs(residual)

        # Quadratic core: |r| ≤ δ
        if abs_r <= delta:
            if residual >= 0:
                # 0 ≤ r ≤ δ
                loss = tau * residual**2 / (2 * delta)
                grad = tau * residual / delta
            else:
                # -δ ≤ r < 0
                loss = (1 - tau) * residual**2 / (2 * delta)
                grad = (1 - tau) * residual / delta
            return loss, grad

        # Linear tails: |r| > δ
        if residual > delta:
            # r > δ : shift tail down by τδ/2 for continuity
            loss = tau * (residual - delta / 2)
            grad = tau
            return loss, grad
        else:  # residual < -δ
            # r < -δ : shift tail down by (1-τ)δ/2 for continuity
            loss = (1 - tau) * (-residual - delta / 2)
            grad = tau - 1
            return loss, grad

    def value(self, y, w, Xw):
        """
        Compute the mean loss across all samples.

        Parameters
        ----------
        y : ndarray, shape (n_samples,)
            Target values

        w : ndarray, shape (n_features,)
            Current coefficient values

        Xw : ndarray, shape (n_samples,)
            Model predictions (X @ w)

        Returns
        -------
        loss : float
            Mean loss value across all samples
        """
        n_samples = len(y)
        res = 0.0
        for i in range(n_samples):
            # Calculate loss for each residual and sum
            loss_i, _ = self._loss_and_grad_scalar(y[i] - Xw[i])
            res += loss_i
        return res / n_samples

    def _dr(self, residual):
        """
        Compute gradient of the loss with respect to residuals (vectorized).

        This function calculates ∂ρ/∂r for an array of residuals.

        Parameters
        ----------
        residual : ndarray
            Array of residuals

        Returns
        -------
        grad : ndarray
            Gradient values for each residual
        """
        tau = self.quantile
        delt = self.delta

        # s(r) = τ for r >= 0, (1-τ) for r < 0
        scale = np.where(residual >= 0, tau, 1 - tau)

        # Inside quadratic zone: grad = scale * (r / delt)
        # Outside quadratic zone: grad = τ for r > δ, (τ-1) for r < -δ
        dr = np.where(
            np.abs(residual) <= delt,
            scale * (residual / delt),  # Quadratic region: r/δ * s(r)
            np.sign(residual) * scale   # Linear regions: ±s(r)
        )
        return dr

    def gradient_scalar(self, X, y, w, Xw, j):
        """
        Compute gradient for a single feature j.

        This is used in coordinate descent algorithms.
        """
        r = y - Xw
        dr = self._dr(r)
        return - X[:, j].dot(dr) / len(y)

    def gradient_scalar_sparse(self, X_data, X_indptr, X_indices, y, Xw, j):
        """Compute gradient for a single feature j with sparse data."""
        r = y - Xw
        dr = self._dr(r)
        idx_start, idx_end = X_indptr[j], X_indptr[j + 1]
        rows = X_indices[idx_start:idx_end]
        vals = X_data[idx_start:idx_end]
        return - np.dot(vals, dr[rows]) / len(y)

    def full_grad_sparse(self, X_data, X_indptr, X_indices, y, Xw):
        """
        Compute the full gradient vector for sparse data.

        This is a more efficient implementation for sparse matrices.
        """
        n_features = len(X_indptr) - 1
        n_samples = len(y)
        grad = np.zeros(n_features, dtype=Xw.dtype)

        # Calculate residuals and their gradients
        for j in range(n_features):
            g = 0.0
            for idx in range(X_indptr[j], X_indptr[j + 1]):
                i = X_indices[idx]
                residual = y[i] - Xw[i]
                _, grad_r = self._loss_and_grad_scalar(residual)
                g += -X_data[idx] * grad_r
            grad[j] = g / n_samples
        return grad

    def intercept_update_step(self, y, Xw):
        """
        Compute the gradient update for the intercept.

        This is used when fitting an intercept separately.
        """
        n_samples = len(y)
        update = 0.0
        for i in range(n_samples):
            residual = y[i] - Xw[i]
            _, grad_r = self._loss_and_grad_scalar(residual)
            update += -grad_r
        return update / n_samples

    def initialize(self, X, y):
        """Initialize any necessary values before optimization (not used)."""
        pass

    def initialize_sparse(self, X_data, X_indptr, X_indices, y):
        """Initialize for sparse data (not used)."""
        pass

    def gradient(self, X, y, Xw):
        """
        Compute the full gradient vector for dense data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix

        y : ndarray, shape (n_samples,)
            Target values

        Xw : ndarray, shape (n_samples,)
            Model predictions

        Returns
        -------
        grad : ndarray, shape (n_features,)
            Gradient vector
        """
        n_samples, n_features = X.shape
        grad = np.zeros(n_features)
        for j in range(n_features):
            grad[j] = self.gradient_scalar(X, y, None, Xw, j)
        return grad
