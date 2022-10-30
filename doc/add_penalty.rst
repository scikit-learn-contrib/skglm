How to add a custom penalty
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A penalty is a jitclass which must inherit from the ``BasePenalty`` class:

.. literalinclude:: ../skglm/penalties/base.py
   :pyobject: BasePenalty


To implement your own penalty, you only need to define a new jitclass, inheriting from ``BasePenalty`` and define how its value, proximal operator, distance to subdifferential (for KKT violation) and penalized features are computed.

A case in point: defining L1 penalty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :math:`\ell_1` penalty is defined as follows:

.. math::
   \lvert\lvert x \rvert\rvert_1 = \sum_{i=1}^p |x_i| \enspace .


The regularization level is controlled by the hyperparameter ``alpha``, that is defined and initialized in the constructor of the class.

The method ``get_spec`` allows to strongly type the attributes of the penalty object, thus allowing Numba to JIT-compile the class, making computations in the methods fast.
This method should return an iterable of tuples, the first element being the name of the attribute, and the second its Numba type.


To optimize an objective with a given penalty, skglm needs at least the proximal operator of the penalty applied to the :math:`j`-th coordinate.
For the ``L1`` penalty, it is the well-known soft-thresholding operator:

.. math::
    \textrm{ST}(x, \lambda) = \mathrm{max}(0, \lvert x \rvert - \lambda) \mathrm{sgn}(x) \enspace .


Note that skglm expects the threshold level to be the regularization hyperparameter ``alpha`` **scaled by** the stepsize.


Besides, by default all solvers in skglm have ``ws_strategy`` turned on to ``subdiff``.
This means that the optimality conditions (thus the stopping criterion) is computed using the method ``subdiff_distance`` of the penalty.
If not implemented, the user should set ``ws_strategy`` on ``fixpoint``.

For the ``L1`` penalty, the distance of the negative gradient of the datafit to the subdifferential of the penalty reads

.. math::
   \mathrm{dist}(-\nabla_j F(\beta), \partial |\beta_j|) = 


When putting all together, this gives the implementation of the ``L1`` penalty:


.. literalinclude:: ../skglm/penalties/separable.py
   :pyobject: L1

