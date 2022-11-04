.. _getting_started:

===============
Getting started
===============
---------------

This page provides a starter example to get familiar with ``skglm`` and explore some of its features.

In the first section, we fit a Lasso estimator on a high dimensional
toy dataset (number of features is largely greater than the number of samples). Linear models doesn't generalize well
for unseen dataset. By adding a penalty, :math:`\ell_1` penalty, we can train estimator that overcome this drawback.

The last section, we explore other combinations of datafit and penalty to create a custom estimator that achieves a lower prediction error,
in the sequel :math:`\ell_1` Huber regression. We show that ``skglm`` is perfectly adapted to these experiments thanks to its modular design.

Beforehand, make sure that you have already installed ``skglm``

.. code-block:: shell

    # using pip
    pip install -U skglm

    # using conda
    conda install skglm

-------------------------


Fitting a Lasso estimator
-------------------------

Let's start first by generating a toy dataset and splitting it to train and test sets.
For that, we will use ``scikit-learn`` 
`make_regression <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html#sklearn.datasets.make_regression>`_

.. code-block:: python

    # imports
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    # generate toy data
    X, y = make_regression(n_samples=100, n_features=1000)
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y)

Then let's fit ``skglm`` :ref:`Lasso <skglm.Lasso>` estimator and prints its score on the test set.

.. code-block:: python

    # import estimator
    from skglm import Lasso
    
    # init and fit
    estimator = Lasso()
    estimator.fit(X_train, y_train)

    # compute R²
    estimator.score(X_test, y_test)


.. note::

    - The first fit after importing ``skglm`` has an overhead as ``skglm`` uses `Numba <https://numba.pydata.org/>`_ 
      The subsequent fits will achieve top speed since Numba compilation is cached.

``skglm`` has several other ``scikit-learn`` compatible estimators.
Check the :ref:`API <Estimators>` for more information about the available estimators.


Fitting :math:`\ell_1` Huber regression
---------------------------------------

Suppose that the latter dataset contains outliers and we would like to mitigate their effects on the learned coefficients
while having an estimator that generalizes well to unseen data. Ideally, we would like to fit a :math:`\ell_1` Huber regressor.

``skglm`` offers high flexibility to compose custom estimators. Through a simple API, it is possible to combine any
``skglm`` :ref:`datafit <Datafits>` and :ref:`penalty <Penalties>`.

.. note::

    - :math:`\ell_1` regularization is not supported in ``scikit-learn`` for HuberRegressor

Let's explore how to achieve that.


Generate corrupt data
*********************

We will use the same script as before except that we will take 10 samples and corrupt their values.

.. code-block:: python

    # imports
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    # generate toy data
    X, y = make_regression(n_samples=100, n_features=1000)

    # select and corrupt 10 random samples
    y[np.random.choice(n_samples, 10)] = 100 * y.max()

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y)


Now let's compose a custom estimator using :ref:`GeneralizedLinearEstimator <skglm.GeneralizedLinearEstimator>`.

.. code-block:: python

    # import penalty and datafit
    from skglm.penalties import L1
    from skglm.datafits import Huber

    # import GLM estimator
    from skglm import GeneralizedLinearEstimator

    # build and fit estimator
    estimator = GeneralizedLinearEstimator(
        Huber(1.),
        L1(alpha=1.)
    )
    estimator.fit(X_train, y_train)


.. note::

    - Here the arguments given to the datafit and penalty are arbitrary and given just for sake of illustration.


It is possible to combine any supported datafit and penalty. Explore the list of supported :ref:`datafits <Datafits>` 
and :ref:`penalties <Penalties>`.

.. important::

    - It is possible to create custom datafit and penalties. Check the tutorials on :ref:`how to add a custom datafit <how_to_add_custom_datafit>` 
      and :ref:`how to add a custom penalty <how_to_add_custom_penalty>`.


Explore further advanced topics and get hands-on examples on the :ref:`tutorials page <tutorials>`