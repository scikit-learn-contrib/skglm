.. _prox_nn_group_lasso:

=============================
Details on the proximity operator of the group Lasso
=============================


This tutorial presents how to derive the proximity operator of the :math:`l_2`-penalty, and the :math:`l_2`-penalty with nonnegative constraints.


Proximity operator of the group Lasso
=============

Let

.. math::
    g:x \mapsto \norm{x}_2
    ,

then

.. math::
    :label: fenchel

    g^{\star}:x \mapsto i_{\norm{x}_2 \leq 1}
    ,

and for all :math:`x \in \mathbb{R}^p`

.. math::
    :label: prox_projection

    \text{prox}_{g^{\star}}(x)
    =
    \text{proj}_{\mathcal{B}_2}(x) = \frac{x}{\max(\norm{x}_2, 1)}
    .

Using the Moreau decomposition and the two equations above one has


.. math::

    \text{prox}_{\lambda g}(x)
    =
    x
    - \lambda \text{prox}_{\lambda g^\star}(x/\lambda)

.. math::

    = x
    - \lambda \text{prox}_{g^\star}(x/\lambda)

.. math::

    = x
    - \lambda  \frac{x/\lambda}{\max(\norm{x/\lambda}_2, 1)}

.. math::

    = x
    - \frac{\lambda x}{\max(\norm{x}_2, \lambda)}

.. math::

    = (1 - \frac{\lambda}{\norm{x}})_{+} x
    .

A similar formula can be derived for the group Lasso with nonnegative constraints.


Proximity operator of the group Lasso with positivity constraints
=============

TODO

References
==========
