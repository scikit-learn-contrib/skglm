:orphan:


How to add a custom datafit
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _how:

Motivated by generalized linear models but not limited to it, skglm solves problems of the form

.. math::
      \hat{\beta} \in
      \arg\min_{\beta \in \mathbb{R}^p}
      F(X\beta) + \Omega(\beta)
      := \sum_{i=1}^n f_i(X_i^{\top}\beta) + \sum_{j=1}^p \Omega_j(\beta_j)
      \enspace .


Here, :math:`X \in \mathbb{R}^{n \times p}` denotes the design matrix with :math:`n` samples and :math:`p` features, and :math:`\beta \in \mathbb{R}^p` is the coefficient vector.

skglm can solve any problems of this form with arbitrary smooth datafit :math:`F` and arbitrary proximable penaty :math:`\Omega`, by defining two classes: a ``Penalty`` and a ``Datafit``.

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

A ``Datafit`` is a jitclass which must inherit from the ``BaseDatafit`` class:

.. literalinclude:: ../skglm/datafits/base.py
   :pyobject: BaseDatafit


To define a custom datafit, you need to implement the methods declared in the ``BaseDatafit`` class (value, gradient, gradient when the design matrix is sparse).
As an example, we show how to implement the Poisson datafit in skglm.


A case in point: defining Poisson datafit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, this requires deriving some useful quantities used by the optimizers like the gradient or the Hessian matrix of the datafit.
With :math:`y \in \mathbb{R}^n` the target vector, the Poisson datafit reads

.. math::
    f(X\beta) = \frac{1}{n}\sum_{i=1}^n \exp(X_i^{\top}\beta) - y_i X_i^{\top}\beta
    \enspace .


Let's define some useful quantities needed in skglm. For :math:`z \in \mathbb{R}^n` and :math:`\beta \in \mathbb{R}^p`,

.. math::
   f(z) = \sum_{i=1}^n f_i(z_i)  \qquad  F(\beta) = f(X\beta)
   \enspace .


Computing the gradient of :math:`F` and its Hessian matrix yields

.. math::
   \nabla F(\beta) = X^{\top} \underbrace{\nabla f(X\beta)}_\textrm{raw grad} \qquad \nabla^2 F(\beta) = X^{\top} \underbrace{\nabla^2 f(X\beta)}_\textrm{raw hessian} X
   \enspace .


Besides, it directly follows that

.. math::
   \nabla f(z) = (f_i'(z_i))_{i \in [n]} \qquad \nabla^2 f(z) = \textrm{diag}(f_i''(z_i))_{i \in [n]}
   \enspace .


Back to the Poisson datafit, following the definition of the datafit, we have

.. math::
    f_i(z_i) = \frac{1}{n} \left(\exp(z_i) - y_iz_i\right)
    \enspace .


Therefore,

.. math::
   f_i'(z_i) = \frac{1}{n}(\exp(z_i) - y_i) \qquad f_i''(z_i) = \frac{1}{n}\exp(z_i)
   \enspace .


Computing ``raw_grad`` and ``raw_hessian`` for the Poisson datafit yields

.. math::
   \nabla f(X\beta) = \frac{1}{n}(\exp(X_i^{\top}\beta) - y_i)_{i \in [n]} \qquad \nabla^2 f(X\beta) = \frac{1}{n}\textrm{diag}(\exp(X_i^{\top}\beta))_{i \in [n]}
   \enspace .


Both ``raw_grad`` and ``raw_hessian`` are methods used by the ``ProxNewton`` solver.

But other optimizers require different methods to be implemented. For instance, ``AndersonCD`` uses the ``gradient_scalar`` method.

The method ``gradient_scalar`` is the derivative of the datafit with respect to the :math:`j`-th coordinate of :math:`\beta`.

For the Poisson datafit, this yields

.. math::
    \frac{\partial F(\beta)}{\partial \beta_j} = \frac{1}{n}
      \sum_{i=1}^n X_{i,j} \left(
         \exp(X_i^{\top}\beta) - y 
      \right)
      \enspace .


When implementing these quantites in the ``Poisson`` datafit class, this gives:

.. literalinclude:: ../skglm/datafits/single_task.py
   :pyobject: Poisson


Note that we have not initialized any quantities in the ``initialize`` method.
Usually it is used to compute a Lipschitz constant of the datafit, used by the optimizers as a step size.
However, in this example, the Poisson datafit has no Lipschitz constant since the eigenvalues of the Hessian matrix are unbounded. 
This implies that a step size is not known in advance and a line search has to be performed at every epoch by the optimizer.
