:orphan:

.. _how:

With skglm, you can solve any custom Generalized Linear Model with arbitrary smooth datafit and arbitrary proximable penalty, by defining two classes: a ``Penalty`` and a ``Datafit``.

They can then be passed to as :class:`skglm.GeneralizedLinearEstimator`, using ``is_classif`` to specify if the task is classification or regression.

.. code-block:: python

   clf = GeneralizedLinearEstimator(
      MyDatafit(), MyPenalty(), is_classif=True, verbose=0)


How to add a custom penalty
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A penalty is a jitclass which must inherit from the ``BasePenalty`` class:

.. literalinclude:: ../skglm/penalties/base.py
   :pyobject: BasePenalty


To implement your own penalty, you only need to define a new jitclass, inheriting from ``BasePenalty`` and define how its value, proximal operator, distance to subdifferential (for KKT violation) and penalized features are computed.

For example, the implementation of the ``L1`` penalty is:

.. literalinclude:: ../skglm/penalties/separable.py
   :pyobject: L1



How to add a custom datafit
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``Datafit`` is a jitclass which must inherit from the ``BaseDatafit`` class:

.. literalinclude:: ../skglm/datafits/base.py
   :pyobject: BaseDatafit


To define a custom datafit, you need to implement the methods declared in the ``BaseDatafit`` class (value, gradient, gradient when the design matrix is sparse).
See for example the implementation of the ``Quadratic`` datafit:

.. literalinclude:: ../skglm/datafits/single_task.py
   :pyobject: Quadratic
