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

Using the Moreau decomposition, Equations :eq:`fenchel` and :eq:`prox_projection`, one has


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
    - \lambda \text{prox}_{\lambda h^\star}(x/\lambda)

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



Subdifferential of the positive Group Lasso penalty
===================================================

For the ```subdiff_diff``` working set strategy, we compute the (distance to the) subdifferential of
the :math:`|| \cdot || + \iota_{\geq 0}`` penalty at a point :math:`w`.
Because the penalty is separable over group, we consider a block of variables, in :math:`\mathbb{R}^g`.

If any component of :math:`w` is strictly negative, the subdifferential is empty, and the distance is :math:`+ \infty`.

At a non zero point with strictly positive entries, the penalty is differentiable with only subgradient :math:`w/ {|| w ||}`.

At :math:`w = 0`, the subdifferential is:

.. math::

    \lambda \partial || \cdot ||_2 + \partial \iota_{\geq 0} = \lambda \mathcal{B}_2 + \mathbb{R}_-^g

where :math:`\mathcal{B}_2`` is the unit ball.

Let :math:`v \in \mathbb{R}^g`, and :math:`\hat v`` its projection onto :math:`\lambda \mathcal{B}_2 + \mathbb{R}_-^g`.
It is clear that for :math:`j` such that :math:`v_j \leq 0`, :math:`v_j = \hat v_j`.
Then, the entries in :math:`\mathcal{S} = \{j : v_j > 0}` are simply given by the projection of :math:`v_\mathcal{S}` onto :math:`\lambda \mathcal{B}_2`.


References
==========

