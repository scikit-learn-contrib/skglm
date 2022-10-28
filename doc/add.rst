:orphan:


Motivated by generalized linear models but not limited to it, skglm solves problems of the form

.. math::
   \begin{equation}
      \hat{\beta} \in
      \arg\min_{\beta \in \mathbb{R}^p}
      F(X\beta) + \Omega(\beta)
      := \sum_{i=1}^n f_i(x_i^{\top}\beta) + \sum_{j=1}^p \Omega_j(\beta_j)
      \enspace .
   \end{equation}

.. _how:

Here, X \in \mathbb{R}^{n \times p} denotes the design matrix with n samples and p features, and \beta \in \mathbb{R}^p is the coefficient vector.

skglm can solve any problems of this form with arbitrary smooth datafit F and arbitrary proximable penaty \Omega, by defining two classes: a ``Penalty`` and a ``Datafit``.

They can then be passed to a :class:`~skglm.GeneralizedLinearEstimator`.


.. code-block:: python

   clf = GeneralizedLinearEstimator(
      MyDatafit(),
      MyPenalty(),
   )


.. How to add a custom penalty
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~
..
.. A penalty is a jitclass which must inherit from the ``BasePenalty`` class:
..
.. .. literalinclude:: ../skglm/penalties/base.py
..    :pyobject: BasePenalty
..
..
.. To implement your own penalty, you only need to define a new jitclass, inheriting from ``BasePenalty`` and define how its value, proximal operator, distance to subdifferential (for KKT violation) and penalized features are computed.
..
.. For example, the implementation of the ``L1`` penalty is:
..
.. .. literalinclude:: ../skglm/penalties/separable.py
..    :pyobject: L1
..


How to add a custom datafit
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``Datafit`` is a jitclass which must inherit from the ``BaseDatafit`` class:

.. literalinclude:: ../skglm/datafits/base.py
   :pyobject: BaseDatafit


To define a custom datafit, you need to implement the methods declared in the ``BaseDatafit`` class (value, gradient, gradient when the design matrix is sparse).

As an example, we show how to implement the Poisson datafit in skglm.

First, this requires deriving some useful quantities used by the optimizers like the gradient or the Hessian matrix of the datafit.

The Poisson datafit reads

.. math::
   \begin{equation}
      F(X\beta) = \sum_{i=1}^n \exp{X_i^{\top}\beta} - y_i X_i^{\top}\beta
   \end{equation}









