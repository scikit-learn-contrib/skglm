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

    def get_params(self, deep=True):
        """Get parameters for this penalty.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this penalty and
            contained subobjects that are penalties.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return self.params_to_dict()

    def set_params(self, **params):
        """Set the parameters of this penalty.

        Parameters
        ----------
        **params : dict
            Penalty parameters.

        Returns
        -------
        self : object
            Returns self.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def value(self, w):
        """Value of penalty at vector w."""

    def is_penalized(self, n_features):
        """Return a binary mask with the penalized features."""

    def generalized_support(self, w):
        """Return a mask which is True for coefficients in the generalized support."""
