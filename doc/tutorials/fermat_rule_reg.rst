.. _fermat_rule_reg:

======================================================
Mathematics Behind L1 Regularization and Fermat's Rule
======================================================

This tutorial presents the mathematics behind solving the optimization problem
:math:`\min f(x) + \lambda \|x\|_1` and demonstrates why the solution is zero when
:math:`\lambda` is greater than the infinity norm of the gradient of :math:`f` at zero, therefore justifying the choice in skglm of

.. code-block::
alpha_max = (popu_X.T @ (1 - popu_Y) / len(popu_Y)).max()

Problem setup
=============

Consider the optimization problem:

.. math::
    \min_x f(x) + \lambda \|x\|_1

where:

- :math:`f: \mathbb{R}^d \to \mathbb{R}` is a differentiable function,
- :math:`\|x\|_1` is the L1 norm of :math:`x`,
- :math:`\lambda \in \mathbb{R}` is a regularization parameter.

We aim to determine the conditions under which the solution to this problem is :math:`x = 0`.

Theoretical Background
======================

According to Fermat's rule, the minimum of the function occurs where the subdifferential of the objective function includes zero. For our problem, the objective function is:

.. math::
    g(x) = f(x) + \lambda \|x\|_1

The subdifferential of :math:`\|x\|_1` at 0 is the L-infinity ball:

.. math::
    \partial \|x\|_1 |_{x=0} = \{ u \in \mathbb{R}^n : \|u\|_{\infty} \leq 1 \}



References
==========

.. _1:
[1] Eugene Ndiaye, Olivier Fercoq, Alexandre Gramfort, and Joseph Salmon. 2017. Gap safe screening rules for sparsity enforcing penalties. J. Mach. Learn. Res. 18, 1 (January 2017), 4671–4703.

