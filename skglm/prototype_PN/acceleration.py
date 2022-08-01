import numpy as np
from numba import float64, int32
from numba.experimental import jitclass

spec_AA = (
    ('K', int32),
    ('current_iter', int32),
    ('arr_w_', float64[:, ::1]),
    ('arr_Xw_', float64[:, ::1])
)


@jitclass(spec_AA)
class JitAndersonAcceleration:
    """Abstraction of Anderson Acceleration.

    Extrapolate the asymptotic VAR ``w`` and ``Xw``
    based on ``K`` previous iterations.

    Parameters
    ----------
    K : int
        Number of previous iterates to consider for extrapolation.
    """

    def __init__(self, K, n_samples, n_features):
        self.K, self.current_iter = K, 0
        self.arr_w_ = np.zeros((n_features, K))
        self.arr_Xw_ = np.zeros((n_samples, K))

    def extrapolate(self, w, Xw):
        """Return w, Xw, and a bool indicating whether they were extrapolated."""
        if self.current_iter <= self.K:
            self.arr_w_[:, self.current_iter] = w
            self.arr_Xw_[:, self.current_iter] = Xw
            self.current_iter += 1
            return w, Xw, False

        U = np.diff(self.arr_w_)  # compute residuals

        # compute extrapolation coefs
        try:
            inv_UTU_ones = np.linalg.solve(U.T @ U, np.ones(self.K))
        except:
            return w, Xw, False
        finally:
            self.current_iter = 0

        # extrapolate
        C = inv_UTU_ones / np.sum(inv_UTU_ones)
        # floating point errors may cause w and Xw to disagree
        return self.arr_w_[:, 1:] @ C, self.arr_Xw_[:, 1:] @ C, True
