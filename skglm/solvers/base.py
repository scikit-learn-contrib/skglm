from abc import abstractmethod


class BaseSolver():
    """Base class for solvers."""

    @abstractmethod
    def solve(self, X, y, datafit, penalty, w_init, Xw_init):
        """Solve an optimization problem.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data.

        y : array, shape (n_samples,)
            Target values.

        datafit : instance of Datafit class
            Datafitting term.

        penalty : instance of Penalty class
            Penalty used in the model.

        w_init : array, shape (n_features,)
            Coefficient vector.

        Xw_init : array, shape (n_samples,)
            Model fit.

        Returns
        -------
        coefs : array, shape (n_features + fit_intercept, n_alphas)
            Coefficients along the path.

        obj_out : array, shape (n_iter,)
            The objective values at every outer iteration.

        stop_crit : float
            Value of stopping criterion at convergence.
        """
