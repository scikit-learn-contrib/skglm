.. _reg_sol_zero:

==========================================================
Critical regularization strength above which solution is 0
==========================================================

This tutorial presents the mathematics behind solving the optimization problem
:math:`\min f(x) + \lambda \|x\|_1` and demonstrates why the solution is zero when
:math:`\lambda` is greater than the infinity norm of the gradient of :math:`f` at zero, therefore justifying the choice in skglm of

.. code-block::
alpha_max = np.max(np.abs(gradient0))

However, the regularization parameter used at the end should preferably be a fraction of this (e.g. `alpha = 0.01 * alpha_max`).

Problem setup
=============

Consider the optimization problem:

.. math::
    \min_x f(x) + \lambda \|x\|_1

where:

- :math:`f: \mathbb{R}^d \to \mathbb{R}` is a convex differentiable function,
- :math:`\|x\|_1` is the L1 norm of :math:`x`,
- :math:`\lambda > 0` is a regularization parameter.

We aim to determine the conditions under which the solution to this problem is :math:`x = 0`.

Theoretical Background
======================

According to Fermat's rule, the minimum of the function occurs where the subdifferential of the objective function includes zero. For our problem, the objective function is:

.. math::
    g(x) = f(x) + \lambda \|x\|_1

The subdifferential of :math:`\|x\|_1` at 0 is the L-infinity ball:

.. math::
    \partial \|x\|_1 |_{x=0} = \{ u \in \mathbb{R}^d : \|u\|_{\infty} \leq 1 \}

Thus, for :math:`0 \in \partial g(x)` at :math:`x=0`:

.. math::
    0 \in \nabla f(0) + \lambda \partial \|x\|_1 |_{x=0}

which implies, given that the dual norm of L1-norm is L-infinity:

.. math::
    \|\nabla f(0)\|_{\infty} \leq \lambda

If :math:`\lambda > \|\nabla f(0)\|_{\infty}`, then the only solution is :math:`x=0`.

Example
=======

Consider the loss function for Ordinary Least Squares :math:`f(x) = \frac{1}{2n} \|Ax - b\|_2^2`, where :math:`n` is the number of samples. We have:

.. math::
    \nabla f(x) = \frac{1}{n}A^T (Ax - b)

At :math:`x=0`:

.. math::
    \nabla f(0) = -\frac{1}{n}A^T b

The infinity norm of the gradient at 0 is:

.. math::
    \|\nabla f(0)\|_{\infty} = \frac{1}{n}\|A^T b\|_{\infty}

For :math:`\lambda \geq \frac{1}{n}\|A^T b\|_{\infty}`, the solution to :math:`\min_x \frac{1}{2n} \|Ax - b\|_2^2 + \lambda \|x\|_1` is :math:`x=0`.



References
==========

Refer to the section 3.1 and proposition 4 in particular of the following article for more details.

.. _1:
[1] Eugene Ndiaye, Olivier Fercoq, Alexandre Gramfort, and Joseph Salmon. 2017. Gap safe screening rules for sparsity enforcing penalties. J. Mach. Learn. Res. 18, 1 (January 2017), 4671–4703.

