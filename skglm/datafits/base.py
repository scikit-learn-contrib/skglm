
class BaseDatafit:
    """Base class for datafits."""

    @staticmethod
    def inverse_link(x):
        """Inverse link function (identity by default).

        Parameters
        ----------
        x : array-like
            Linear predictor values.

        Returns
        -------
        array-like
            Transformed values in response scale.
        """
        return x

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

    def initialize(self, X, y):
        """Pre-computations before fitting on X and y.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        y : array, shape (n_samples,)
            Target vector.
        """

    def initialize_sparse(self, X_data, X_indptr, X_indices, y):
        """Pre-computations before fitting on X and y when X is a sparse matrix.

        Parameters
        ----------
        X_data : array, shape (n_elements,)
            `data` attribute of the sparse CSC matrix X.

        X_indptr : array, shape (n_features + 1,)
            `indptr` attribute of the sparse CSC matrix X.

        X_indices : array, shape (n_elements,)
            `indices` attribute of the sparse CSC matrix X.

        y : array, shape (n_samples,)
            Target vector.
        """

    def value(self, y, w, Xw):
        """Value of datafit at vector w.

        Parameters
        ----------
        y : array_like, shape (n_samples,)
            Target vector.

        w : array_like, shape (n_features,)
            Coefficient vector.

        Xw: array_like, shape (n_samples,)
            Model fit.

        Returns
        -------
        value : float
            The datafit value at vector w.
        """


class BaseMultitaskDatafit:
    """Base class for multitask datafits."""

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

    def initialize(self, X, Y):
        """Store useful values before fitting on X and Y.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        Y : array, shape (n_samples, n_tasks)
            Multitask target.
        """

    def initialize_sparse(self, X_data, X_indptr, X_indices, Y):
        """Store useful values before fitting on X and Y, when X is sparse.

        Parameters
        ----------
        X_data : array-like
            `data` attribute of the sparse CSC matrix X.

        X_indptr : array-like
            `indptr` attribute of the sparse CSC matrix X.

        X_indices : array-like
            `indices` attribute of the sparse CSC matrix X.

        Y : array, shape (n_samples, n_tasks)
            Target matrix.
        """

    def value(self, Y, W, XW):
        """Value of datafit at matrix W.

        Parameters
        ----------
        Y : array_like, shape (n_samples, n_tasks)
            Target matrix.

        W : array_like, shape (n_features, n_tasks)
            Coefficient matrix.

        XW: array_like, shape (n_samples, n_tasks)
            Model fit.

        Returns
        -------
        value : float
            The datafit value evaluated at matrix W.
        """
