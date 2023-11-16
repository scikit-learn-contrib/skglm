from abc import abstractmethod, ABC
from skglm.utils.validation import check_obj_solver_attr


class BaseSolver(ABC):
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

    def custom_compatibility_check(self, X, y, datafit, penalty):
        """Ensure the solver is suited for the `datafit` + `penalty` problem.

        Parameters
        ----------
        datafit : instance of BaseDatafit
            Datafit.

        penalty : instance of BasePenalty
            Penalty.
        """

    def __call__(self, X, y, datafit, penalty, w_init=None, Xw_init=None):
        """Solve the optimization problem after validating its compatibility.

        A proxy of ``solve`` method that implicitly ensures the compatibility
        of datafit and penalty with the solver.

        Examples
        --------
        >>> ...
        >>> coefs, obj_out, stop_crit = solver(X, y, datafit, penalty)
        """
        self._validate(datafit, penalty)
        self.solve(X, y, datafit, penalty, w_init, Xw_init)

    def _validate(self, X, y, datafit, penalty):
        # execute both attributes checks and `custom_compatibility_check`
        check_obj_solver_attr(datafit, self, self._datafit_required_attr)
        check_obj_solver_attr(datafit, self, self._penalty_required_attr)

        self.custom_compatibility_check(datafit, penalty)
