.. _getting_started:

===============
Getting started
===============

This page provides a started examples to get familiar with ``sglm``
and explore some of its features.

We start by fitting a Lasso estimator on a toy dataset using ``skglm`` Lasso estimator.
and then we explore how modularity of the package by building and fitting a :math:`\ell_1` Huber regression.

Beforehand, make sure that you have already installed ``skglm`` either through

.. code-block:: shell

    $ pip install -U skglm

or

.. code-block:: shell

    $ conda install skglm

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

Then let's fit ``skglm`` :ref:`Lasso <skglm.Lasso>` estimator and prints it's score on the test set.

.. code-block:: python

    # import estimator
    from skglm import Lasso
    
    # init and fit
    estimator = Lasso()
    estimator.fit(X_train, y_train)

    # compute R²
    estimator.score(X_test, y_test)


.. note::

    - Every first fit of ``skglm`` estimator take some extra time as ``skglm`` uses `Numba <https://numba.pydata.org/>`_ 
      to just-in-time compile the code.
      Afterward, the other fits will be super fast.

``skglm`` has several other ``scikit-learn`` compatible estimators.
Check the :ref:`API <Estimators>` for more information about the available estimators.

