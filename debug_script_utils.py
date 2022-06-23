import numpy as np


class AA_singular:
    """Abstraction of Anderson Acceleration.

    Extrapolate the asymptotic VAR ``w`` and ``Xw``
    based on ``K`` previous iterations.

    Parameters
    ----------
    K : int
        Number of previous iterates to consider for extrapolation.
    """

    def __init__(self, K):
        self.K, self.current_iter = K, 0
        self.arr_w_, self.arr_Xw_ = None, None

    def extrapolate(self, w, Xw):
        """Return ``w``, ``Xw``, and a bool indicating wether they were extrapolated."""
        if self.arr_w_ is None or self.arr_Xw_ is None:
            self.arr_w_ = np.zeros((w.shape[0], self.K+1))
            self.arr_Xw_ = np.zeros((Xw.shape[0], self.K+1))

        if self.current_iter <= self.K:
            self.arr_w_[:, self.current_iter] = w
            self.arr_Xw_[:, self.current_iter] = Xw
            self.current_iter += 1
            return w, Xw

        U = np.diff(self.arr_w_, axis=1)  # compute residuals

        # compute extrapolation coefs
        inv_UTU_ones, n_K_excluded = self.solve_UTU_ones(U.T @ U)
        C = inv_UTU_ones / np.sum(inv_UTU_ones)

        # extrapolate
        w_acc = self.arr_w_[:, 1+n_K_excluded:] @ C
        Xw_acc = self.arr_Xw_[:, 1+n_K_excluded:] @ C

        self.current_iter = 0  # reset counter
        return w_acc, Xw_acc

    def solve_UTU_ones(self, UTU):
        """Solve ``UTU = I`` while handing the singularity case.

        Returns solution and number of excluded iters.
        """
        dim_UTU = UTU.shape[0]

        if dim_UTU == 1 and UTU[0, 0] == 0:
            return np.ones(1), self.K - 1

        try:
            inv_UTU_ones = np.linalg.solve(UTU, np.ones(dim_UTU))
        except np.linalg.LinAlgError:
            return self.solve_UTU_ones(UTU[:-1, :-1])

        return inv_UTU_ones, self.K - dim_UTU


class AA_usual:
    """Abstraction of Anderson Acceleration.

    Extrapolate the asymptotic VAR ``w`` and ``Xw``
    based on ``K`` previous iterations.

    Parameters
    ----------
    K : int
        Number of previous iterates to consider for extrapolation.
    """

    def __init__(self, K):
        self.K, self.current_iter = K, 0
        self.arr_w_, self.arr_Xw_ = None, None

    def extrapolate(self, w, Xw):
        """Return ``w`` and ``Xw`` extrapolated."""
        if self.arr_w_ is None or self.arr_Xw_ is None:
            self.arr_w_ = np.zeros((w.shape[0], self.K+1))
            self.arr_Xw_ = np.zeros((Xw.shape[0], self.K+1))

        if self.current_iter <= self.K:
            self.arr_w_[:, self.current_iter] = w
            self.arr_Xw_[:, self.current_iter] = Xw
            self.current_iter += 1
            return w, Xw

        U = np.diff(self.arr_w_, axis=1)  # compute residuals

        # compute extrapolation coefs
        try:
            inv_UTU_ones = np.linalg.solve(U.T @ U, np.ones(self.K))
        except np.linalg.LinAlgError:
            return w, Xw
        finally:
            self.current_iter = 0

        # extrapolate
        C = inv_UTU_ones / np.sum(inv_UTU_ones)
        # floating point errors may cause w and Xw to disagree
        return self.arr_w_[:, 1:] @ C, self.arr_Xw_[:, 1:] @ C
