.. _fermat_rule_reg:

======================================================
Mathematics Behind L1 Regularization and Fermat's Rule
======================================================

This tutorial presents the mathematics behind solving the optimization problem
:math:`\min f(x) + \lambda \|x\|_1` and demonstrates why the solution is zero when
:math:`\lambda` is greater than the infinity norm of the gradient of :math:`f` at zero, therefore justifying the choice in skglm of

.. code-block::
alpha_max = np.max(np.abs(gradient0))

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

Thus, for :math:`0 \in \partial g(x)` at :math:`x=0`:

.. math::
    0 \in \nabla f(0) + \lambda \partial \|x\|_1 |_{x=0}

which implies, given that the dual of L1-norm is L-infinity:

.. math::
    \|\nabla f(0)\|_{\infty} \leq \lambda

If :math:`\lambda > \|\nabla f(0)\|_{\infty}`, then the only solution is :math:`x=0`.

Example
=======

Consider the loss function for Ordinary Least Squares :math:`f(x) = \frac{1}{2} \|Ax - b\|_2^2`. We have:

.. math::
    \nabla f(x) = A^T (Ax - b)

At :math:`x=0`:

.. math::
    \nabla f(0) = -A^T b

The infinity norm of the gradient at 0 is:

.. math::
    \|\nabla f(0)\|_{\infty} = \|A^T b\|_{\infty}

For :math:`\lambda > \|A^T b\|_{\infty}`, the solution to :math:`\min_x \frac{1}{2} \|Ax - b\|_2^2 + \lambda \|x\|_1` is :math:`x=0`.



References
==========

The first 5 pages of the following article provide sufficient context for the problem at hand.

.. _1:
[1] Eugene Ndiaye, Olivier Fercoq, Alexandre Gramfort, and Joseph Salmon. 2017. Gap safe screening rules for sparsity enforcing penalties. J. Mach. Learn. Res. 18, 1 (January 2017), 4671–4703.

