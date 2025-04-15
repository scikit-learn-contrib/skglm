import warnings
from abc import abstractmethod, ABC

import numpy as np

from skglm.utils.validation import check_attrs
from skglm.utils.jit_compilation import compiled_clone


class BaseSolver(ABC):
    """Base class for solvers.

    Attributes
    ----------
    _datafit_required_attr : list
        List of attributes that must be implemented in Datafit.

    _penalty_required_attr : list
        List of attributes that must be implemented in Penalty.

    Notes
    -----
    For required attributes, if an attribute is given as a list of attributes
    it means at least one of them should be implemented.
    For instance, if

        _datafit_required_attr = (
            "get_global_lipschitz",
            ("gradient", "gradient_scalar")
        )

    it mean datafit must implement the methods ``get_global_lipschitz``
    and (``gradient`` or ``gradient_scaler``).
    """

    _datafit_required_attr: list
    _penalty_required_attr: list

    @abstractmethod
    def _solve(self, X, y, datafit, penalty, w_init, Xw_init):
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

    def custom_checks(self, X, y, datafit, penalty):
        """Ensure the solver is suited for the `datafit` + `penalty` problem.

        This method includes extra checks to perform
        aside from checking attributes compatibility.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data.

        y : array, shape (n_samples,)
            Target values.

        datafit : instance of BaseDatafit
            Datafit.

        penalty : instance of BasePenalty
            Penalty.
        """
        pass

    def solve(self, X, y, datafit, penalty, w_init=None, Xw_init=None,
              *, run_checks=True):
        """Solve the optimization problem after validating its compatibility.

        A proxy of ``_solve`` method that implicitly ensures the compatibility
        of ``datafit`` and ``penalty`` with the solver.

        Examples
        --------
        >>> ...
        >>> coefs, obj_out, stop_crit = solver.solve(X, y, datafit, penalty)
        """
        if "jitclass" in str(type(datafit)):
            warnings.warn(
                "Do not pass a compiled datafit, compilation is done inside solver now")
        else:
            datafit = compiled_clone(datafit, to_float32=X.dtype == np.float32)
            penalty = compiled_clone(penalty)
            # TODO add support for bool spec in compiled_clone
            # penalty = compiled_clone(penalty, to_float32=X.dtype == np.float32)

        if run_checks:
            self._validate(X, y, datafit, penalty)

        return self._solve(X, y, datafit, penalty, w_init, Xw_init)

    def _validate(self, X, y, datafit, penalty):
        # execute: `custom_checks` then check attributes
        self.custom_checks(X, y, datafit, penalty)

        # do not check for sparse support here, make the check at the solver level
        # some solvers like ProxNewton don't require methods for sparse support
        check_attrs(datafit, self, self._datafit_required_attr)
        check_attrs(penalty, self, self._penalty_required_attr)
