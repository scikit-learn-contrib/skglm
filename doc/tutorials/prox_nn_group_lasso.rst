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
Since the penalty is group-separable, we reduce the case where :math:`w` is a block of variables in :math:`\mathbb{R}^g`.

Case :math:`w \notin \mathbb{R}_+^g`
------------------------------------

If any component of :math:`w` is strictly negative, the subdifferential is empty, and the distance is :math:`+ \infty`.

.. math::

    D(v) = + \infty, \quad \forall v \in \mathbb{R}^g
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

where :math:`v^+` is :math:`v` restricted to its positive coordinates.
Intuitively, it is clear that if :math:`v_i < 0`, we can cancel it exactly in the objective function by taking :math:`n_i = - v_i` and :math:`u_i = 0`; on the other hand, if :math:`v_i>0`, taking a non zero :math:`n_i` will only increase the quantity that :math:`u_i` needs to bring closer to 0.

For a rigorous derivation of this, introduce the Lagrangian on a squared objective

.. math::

    \mathcal{L}(u, n, \nu, \mu) =
    \frac{1}{2}\norm{u + n - v}^2 + \nu(\frac{1}{2} \norm{u}^2 - \lambda^2 / 2) + \langle \mu, n \rangle
    ,

and write down the optimality condition with respect to :math:`u` and :math:`n`.
Treat the case :math:`nu = 0` separately; in the other case show that :\math:`u` must be positive, and that :math:`v = (1 + \nu) u + n`, together with :math:`u = \mu / \nu` and complementary slackness, to reach the conclusion.

Case :math:`|| w  || \ne 0`
---------------------------
The subdifferential in that case is :math:`\lambda w / {|| w ||} + C_1 \times \ldots \times C_g` where :math:`C_j = {0}` if :math:`w_j > 0` and :math:`C_j = mathbb{R}_-` otherwise (:math:`w_j =0`).

By letting :math:`p` denotes the projection of :math:`v` onto this set,
one has

.. math::

    p_j = \lambda \frac{w_j}{||w||}  \text{ if }  w_j > 0

and

.. math::

    p_j = \min(v_j, 0)  \text{ otherwise}.

The distance to the subdifferential is then:

.. math::

    D(v) = || v - p || = \sqrt{\sum_{j, w_j > 0} (v_j - \lambda \frac{w_j}{||w||})^2 + \sum_{j, w_j=0} \max(0, v_j)^2

since :math:`v_j - \min(v_j, 0) = v_j + \max(-v_j, 0) = \max(0, v_j)`.



References
==========

[1] `<https://math.stackexchange.com/a/2887332/167258>`_
