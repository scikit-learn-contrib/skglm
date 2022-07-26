from abc import abstractmethod


class BaseDatafit():
    """Base class for datafits."""

    @abstractmethod
    def initialize(self, X, y):
        """Pre-computations before fitting on X and y.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        y : array, shape (n_samples,)
            Target vector.
        """

    @abstractmethod
    def initialize_sparse(
            self, X_data, X_indptr, X_indices, y):
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

    @abstractmethod
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

    @abstractmethod
    def gradient_scalar(self, X, y, w, Xw, j):
        """Gradient with respect to j-th coordinate of w.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        y : array, shape (n_samples,)
            Target vector.

        w : array, shape (n_features,)
            Coefficient vector.

        Xw : array, shape (n_samples,)
            Model fit.

        j : int
            The coordinate at which the gradient is evaluated.

        Returns
        -------
        gradient : float
            The gradient of the datafit with respect to the j-th coordinate of w.
        """

    @abstractmethod
    def gradient_scalar_sparse(self, X_data, X_indptr, X_indices, y, Xw, j):
        """Gradient with respect to j-th coordinate of w when X is sparse.

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

        Xw: array, shape (n_samples,)
            Model fit.

        j : int
            The dimension along which the gradient is evaluated.

        Returns
        -------
        gradient : float
            The gradient of the datafit with respect to the j-th coordinate of w.
        """


class BaseMultitaskDatafit():
    """Base class for multitask datafits."""

    @abstractmethod
    def initialize(self, X, Y):
        """Store useful values before fitting on X and Y.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        Y : array, shape (n_samples, n_tasks)
            Multitask target.
        """

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def gradient_j(self, X, Y, W, XW, j):
        """Gradient with respect to j-th coordinate of W.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Design matrix.

        Y : array, shape (n_samples, n_tasks)
            Target matrix.

        W : array, shape (n_features, n_tasks)
            Coefficient matrix.

        XW : array, shape (n_samples, n_tasks)
            Model fit.

        j : int
            The coordinate along which the gradient is evaluated.

        Returns
        -------
        gradient : array, shape (n_tasks,)
            The gradient of the datafit with respect to the j-th coordinate of W.
        """

    @abstractmethod
    def gradient_j_sparse(self, X_data, X_indptr, X_indices, Y, XW, j):
        """Gradient with respect to j-th coordinate of W when X is sparse.

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

        XW : array, shape (n_samples, n_tasks)
            Model fit.

        j : int
            The coordinate along which the gradient is evaluated.

        Returns
        -------
        gradient : array, shape (n_tasks,)
            The gradient of the datafit with respect to the j-th coordinate of W.
        """
