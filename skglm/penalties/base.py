from numba import float64
from abc import abstractmethod


class BasePenalty():
    """Base class for penalty subclasses."""

    @abstractmethod
    def get_spec(self):
        """Specify the numba types of the class attributes.

        Returns
        -------
        spec: Tuple of (attribute_name, dtype)
            spec to be passed to Numba jitclass to compile the class.
        """

    @abstractmethod
    def params_to_dict(self):
        """Get the parameters to initialize an instance of the class.

        Returns
        -------
        dict_of_params : dict
            The parameters to instantiate an object of the class.
        """

    @abstractmethod
    def value(self, w):
        """Value of penalty at vector w."""

    @abstractmethod
    def prox_1d(self, value, stepsize, j):
        """Proximal operator of penalty for feature j."""

    @abstractmethod
    def subdiff_distance(self, w, grad, ws):
        """Distance of negative gradient to subdifferential at w for features in `ws`.

        Parameters
        ----------
        w: array, shape (n_features,)
            Coefficient vector.

        grad: array, shape (ws.shape[0],)
            Gradient of the datafit at w, restricted to features in `ws`.

        ws: array, shape (ws_size,)
            Indices of features in the working set.

        Returns
        -------
        distances: array, shape (ws.shape[0],)
            The distances to the subdifferential.
        """

    @abstractmethod
    def is_penalized(self, n_features):
        """Return a binary mask with the penalized features."""

    @abstractmethod
    def generalized_support(self, w):
        r"""Return a mask which is True for coefficients in the generalized support."""


def overload_with_l2(klass):
    """Decorate a penalty class to add L2 regularization.

    The resulting penalty reads

    .. math::

        "penalty"(w) + "l2"_"regularization" xx ||w||**2 / 2

    Parameters
    ----------
    klass : Penalty class
        The penalty class to be overloaded with L2 regularization.

    Returns
    -------
    klass : Penalty class
        Penalty overloaded with L2 regularization.
    """
    # keep ref to original methods
    cls_constructor = klass.__init__
    cls_prox_1d = klass.prox_1d
    cls_value = klass.value
    cls_subdiff_distance = klass. subdiff_distance
    cls_params_to_dict = klass.params_to_dict
    cls_get_spec = klass.get_spec

    # implement new methods
    def __init__(self, *args, l2_regularization=0., **kwargs):
        cls_constructor(self, *args, **kwargs)
        self.l2_regularization = l2_regularization

    def prox_1d(self, value, stepsize, j):
        if self.l2_regularization == 0.:
            return cls_prox_1d(self, value, stepsize, j)

        scale = 1 + stepsize * self.l2_regularization
        return cls_prox_1d(self, value / scale, stepsize / scale, j)

    def value(self, w):
        l2_regularization = self.l2_regularization
        if l2_regularization == 0.:
            return cls_value(self, w)

        return cls_value(self, w) + l2_regularization * 0.5 * w ** 2

    def subdiff_distance(self, w, grad, ws):
        if self.l2_regularization == 0.:
            return cls_subdiff_distance(self, w, grad, ws)

        return cls_subdiff_distance(self, w, grad + self.l2_regularization * w[ws], ws)

    def get_spec(self):
        return (('l2_regularization', float64), *cls_get_spec(self))

    def params_to_dict(self):
        return dict(l2_regularization=self.l2_regularization,
                    **cls_params_to_dict(self))

    # override methods
    klass.__init__ = __init__
    klass.value = value
    klass.prox_1d = prox_1d
    klass.subdiff_distance = subdiff_distance
    klass.get_spec = get_spec
    klass.params_to_dict = params_to_dict

    return klass
