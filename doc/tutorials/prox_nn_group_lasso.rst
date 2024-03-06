.. _prox_nn_group_lasso:

===================================
Details on the Positive Group Lasso
===================================

This tutorial presents how to derive the proximity operator and subdifferential of the :math:`l_2`-penalty, and the :math:`l_2`-penalty with nonnegative constraints.


Proximity operator of the group Lasso
=====================================

Let

.. math::
    g:x \mapsto \norm{x}_2
    ,

then its Fenchel-Legendre conjugate is

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

Using the Moreau decomposition, Equations :eq:`fenchel` and :eq:`prox_projection`, one has


.. math::

    \text{prox}_{\lambda g}(x)
    =
    x
    - \lambda \text{prox}_{g^\star/\lambda }(x/\lambda)

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
=================================================================

Let

.. math::
    h:x \mapsto \norm{x}_2
    + i_{x \geq 0}
    .

Let :math:`x \in \mathbb{R}^p` and :math:`S =  \{ j \in 1, ..., p | x_j > 0 \} \in \mathbb{R}^p`, then


.. math::
    :label: fenchel_nn

    h^{\star} :x  \mapsto i_{\norm{x_S}_2 \leq 1}
    ,

and

.. math::
    :label: prox_projection_nn_Sc

    \text{prox}_{h^{\star}}(x)_{S^c}
    =
    x_{S^c}


.. math::
    :label: prox_projection_nn_S

    \text{prox}_{h^{\star}}(x)_S
    =
    \text{proj}_{\mathcal{B}_2}(x_S) = \frac{x_S}{\max(\norm{x_S}_2, 1)}
    .

As before, using the Moreau decomposition and Equation :eq:`fenchel_nn` yields


.. math::

    \text{prox}_{\lambda h}(x)
    =
    x
    - \lambda \text{prox}_{h^\star / \lambda }(x/\lambda)

.. math::

    = x
    - \lambda \text{prox}_{h^\star}(x/\lambda)
    ,

and thus, combined with Equations :eq:`prox_projection_nn_Sc` and :eq:`prox_projection_nn_S` it leads to

.. math::

    \text{prox}_{\lambda h}(x)_{S^c} = 0

.. math::

    \text{prox}_{\lambda h}(x)_{S}
    =
    (1 - \frac{\lambda}{\norm{x_S}})_{+} x_S
    .



.. _subdiff_positive_group_lasso:
Subdifferential of the positive Group Lasso penalty
===================================================

For the ``subdiff_diff`` working set strategy, we compute the distance :math:`D(v)` for some :math:`v` to the subdifferential of the :math:`h` penalty at a point :math:`w`.
Since the penalty is group-separable, we consider a block of variables, in :math:`\mathbb{R}^g`.

Case :math:`w` has a strictly negative coordinate
-------------------------------------------------

If any component of :math:`w` is strictly negative, the subdifferential is empty, and the distance is :math:`+ \infty`.

.. math::

    D(v) = + \infty, \quad \forall v \in \mathbb{R}^g
    .


Case :math:`w` is strictly positive
-----------------------------------

At a non zero point with strictly positive entries, the penalty is differentiable hence its subgradient is the singleton :math:`w / {|| w ||}`.

.. math::

    D(v) = || v - w / {|| w ||} ||, \quad \forall v \in \mathbb{R}^g
    .

Case :math:`w = 0`
------------------

At :math:`w = 0`, the subdifferential is:

.. math::

    \lambda \partial || \cdot ||_2 + \partial \iota_{x \geq 0} = \lambda \mathcal{B}_2 + \mathbb{R}_-^g
    ,

where :math:`\mathcal{B}_2` is the unit ball.

Therefore, the distance to the subdifferential writes

.. math::

    D(v) = \min_{u \in \lambda \mathcal{B}_2, n \in \mathbb{R}_{-}^g} \ || u + n - v ||
    .

Minimizing over :math:`n` then over :math:`u`, thanks to [`1 <https://math.stackexchange.com/a/2887332/167258>`_], yields

.. math::

    D(v) = \max(0, ||v^+|| - \lambda)
    ,

Where :math:`v^+` is :math:`v` restricted to its positive coordinates.


References
==========

[1] `<https://math.stackexchange.com/a/2887332/167258>`_