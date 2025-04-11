
class BasePenalty:
    """Base class for penalty subclasses."""

    def get_spec(self):
        """Specify the numba types of the class attributes.

        Returns
        -------
        spec: Tuple of (attribute_name, dtype)
            spec to be passed to Numba jitclass to compile the class.
        """

    def params_to_dict(self):
        """Get the parameters to initialize an instance of the class.

        Returns
        -------
        dict_of_params : dict
            The parameters to instantiate an object of the class.
        """

    def value(self, w):
        """Value of penalty at vector w."""

    def is_penalized(self, n_features):
        """Return a binary mask with the penalized features."""

    def generalized_support(self, w):
        """Return a mask which is True for coefficients in the generalized support."""
