from abc import abstractmethod
from skglm.utils.validation import check_obj_solver_attr_compatibility


class BaseSolver():
    """Base class for solvers.

    Attributes
    ----------
    _datafit_required_attr : list of str
        List of attributes that must implemented in Datafit.

    _penalty_required_attr : list of str
        List of attributes that must implemented in Penalty.
    """

    _datafit_required_attr: list
    _penalty_required_attr: list

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

    @abstractmethod
    def validate(self, datafit, penalty):
        """Ensure the solver is suited for the `datafit` + `penalty` problem.

        Parameters
        ----------
        datafit : instance of BaseDatafit
            Datafit.

        penalty : instance of BasePenalty
            Penalty.
        """

    def __call__(self, X, y, datafit, penalty, w_init, Xw_init, **kwargs):
        check_obj_solver_attr_compatibility(datafit, self, self._datafit_required_attr)
        check_obj_solver_attr_compatibility(datafit, self, self._penalty_required_attr)

        self.validate(datafit, penalty)

        self.solve(X, y, datafit, penalty, w_init, Xw_init, **kwargs)
