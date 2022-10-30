How to add a custom penalty
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A penalty is a jitclass which must inherit from the ``BasePenalty`` class:

.. literalinclude:: ../skglm/penalties/base.py
   :pyobject: BasePenalty


To implement your own penalty, you only need to define a new jitclass, inheriting from ``BasePenalty`` and define how its value, proximal operator, distance to subdifferential (for KKT violation) and penalized features are computed.

For example, the implementation of the ``L1`` penalty is:

.. literalinclude:: ../skglm/penalties/separable.py
   :pyobject: L1

