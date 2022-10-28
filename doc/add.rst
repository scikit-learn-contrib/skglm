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
      \enspace .
   \end{equation}

Let define some useful quantities needed in skglm.

For z in \mathbb{R}^n,

.. math::
    \begin{equation}
      f(z) = \sum_{i=1}^n f_i(z_i)
    \end{equation}

and \beta \in \mathbb{R}^p,

.. math::
   \begin{equation}
      F(\beta) = f(X\beta)
   \end{equation}


Computing the gradient of F yields

.. math::
   \begin{equation}
      \nabla F(\beta) = X^{\top} \underbrace{\nabla f(X\beta)}_\textrm{raw grad}
       \enspace ,
   \end{equation}

and its Hessian matrix

.. math::
   \begin{equation}
      \nabla^2 F(\beta) = X^{\top} \underbrace{\nabla^2 f(X\beta)}_\textrm{raw hessian} X
    \end{equation}


Besides, it directly follows that:

.. math::
   \begin{equation}
      \nabla f(z) = (f_i'(z_i))_{i \in [n]}
    \end{equation}

and

.. math::
   \begin{equation}
      \nabla^2 f(z) = \textrm{diag}(f_i''(z_i))_{i \in [n]}
      \enspace .
   \end{equation}

Back to the Poisson datafit, following the definition of the datafit, we have

.. math::
   \begin{equation}
      f_i(z_i) = (\exp{z_i} - y_iz_i)
      \enspace .
   \end{equation}

Therefore,

.. math::
   \begin{equation}
      f_i'(z_i) = \exp{z_i} - y_i
   \end{equation}

and,

.. math::
   \begin{equation}
      f_i''(z_i) = \exp{z_i}
      \enspace .
   \end{equation}


Note that for the Poisson datafit, there is no Lipschitz constant since the eigenvalues of the Hessian matrix are unbounded.

This implies that a step size is not known in advance and a line search has to be performed at every step by the optimizer to get a suitable step size.

Computing ``raw_grad`` and ``raw_hessian`` for the Poisson datafit yields

.. math::
   \begin{equation}
      \nabla f(X\beta) = (\exp{X_i^{\top}\beta})_{i \in [n]}
   \end{equation}

and

.. math::
   \begin{equation}
      \nabla^2 f(X\beta) = \textrm{diag}(\exp{X_i^{\top}\beta})_{i \in [n]}
      \enspace .
   \end{equation}


Both ``raw_grad`` and ``raw_hessian`` are methods used by the ``ProxNewton`` solver.

But other datafits are optimized using ``AndersonCD``, which requires the ``gradient_scalar`` method to be implemented.

The method ``gradient_scalar`` is the derivative of the datafit with respect to the j-th coordinate of \beta.

For the Poisson datafit, this yields

.. math::
   \begin{equation}
      \frac{partial F(\beta)}{\beta_j} =
      \frac{1}{n}
      \sum_{i=1}^n X_{:j} \left(
         \exp{Xw} - y 
      \right)
      \enspace .
    \end{equation}


When implementing these quantites in the ``Poisson`` datafit class, this gives:

.. literalinclude:: ../skglm/datafits/single_task.py
   :pyobject: Poisson


