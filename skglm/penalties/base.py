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
        """Get the parameters to initials an instance of the class.

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
