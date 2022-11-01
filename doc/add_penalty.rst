:orphan:

How to add a custom penalty
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _how:

skglm supports any arbitrary proximable penalty.


It is implemented as a jitclass which must inherit from the ``BasePenalty`` class:

.. literalinclude:: ../skglm/penalties/base.py
   :pyobject: BasePenalty

To implement your own penalty, you only need to define a new jitclass, inheriting from ``BasePenalty`` and define how its value, proximal operator, distance to subdifferential (for KKT violation) and penalized features are computed.

A case in point: defining L1 penalty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We detail how the :math:`\ell_1` penalty is implemented in skglm.
For a vector :math:`\beta \in \mathbb{R}^p`, the :math:`\ell_1` penalty is defined as follows:

.. math::
   \lvert\lvert \beta \rvert\rvert_1 = \sum_{i=1}^p |\beta _i| \enspace .


The regularization level is controlled by the hyperparameter :math:`\lambda \in \mathbb{R}^+`, that is defined and initialized in the constructor of the class.

The method ``get_spec`` allows to strongly type the attributes of the penalty object, thus allowing Numba to JIT-compile the class.
It should return an iterable of tuples, the first element being the name of the attribute, the second its Numba type (e.g. `float64`, `bool_`).
Additionally, a penalty should implement ``params_to_dict``, a helper method to get all the parameters of a penalty returned in a dictionary.

To optimize an objective with a given penalty, skglm needs at least the proximal operator of the penalty applied to the :math:`j`-th coordinate.
For the ``L1`` penalty, it is the well-known soft-thresholding operator:

.. math::
    \textrm{ST}(\beta , \lambda) = \mathrm{max}(0, \lvert \beta \rvert - \lambda) \mathrm{sgn}(\beta) \enspace .


Note that skglm expects the threshold level to be the regularization hyperparameter :math:`\lambda \in \mathbb{R}^+` **scaled by** the stepsize.


Besides, by default all solvers in skglm have ``ws_strategy`` turned on to ``subdiff``.
This means that the optimality conditions (thus the stopping criterion) is computed using the method ``subdiff_distance`` of the penalty.
If not implemented, the user should set ``ws_strategy`` to ``fixpoint``.

For the :math:`\ell_1` penalty, the distance of the negative gradient of the datafit :math:`F` to the subdifferential of the penalty reads

.. math::
   \mathrm{dist}(-\nabla_j F(\beta), \partial |\beta_j|) = \begin{cases}
        \mathrm{max}(0, \lvert -\nabla_j F(\beta) \rvert - \lambda) \\
        \lvert -\nabla_j F(\beta) - \lambda \mathrm{sgn}(\beta_j) \lvert \\
    \end{cases}
   \enspace .


The method ``is_penalized`` returns a binary mask with the penalized features.
For the :math:`\ell_1` penalty, all the coefficients are penalized.
Finally, ``generalized_support`` returns the generalized support of the penalty for some coefficient vector ``w``.
It is typically the non-zero coefficients of the solution vector for :math:`\ell_1`.


Optionally, a penalty might implement ``alpha_max`` which returns the smallest :math:`\lambda` for which the optimal solution is a null vector.
Note that since ``lambda`` is a reserved keyword in Python, ``alpha`` in skglm codebase corresponds to :math:`\lambda`.

When putting all together, this gives the implementation of the ``L1`` penalty:


.. literalinclude:: ../skglm/penalties/separable.py
   :pyobject: L1

