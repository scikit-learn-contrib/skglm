.. _alpha_max:

.. meta::
   :description: Tutorial explaining the critical regularization strength (alpha_max) in skglm. Learn conditions for zero solutions in L1-regularized optimization problems.

==========================================================
Critical Regularization Strength above which Solution is 0
==========================================================

This tutorial shows that for :math:`\lambda \geq \lambda_{\text{max}} = || \nabla f(0) ||_{\infty}`, the solution to
:math:`\min f(x) + \lambda || x ||_1` is 0.

In skglm, we thus frequently use

.. code-block::

    alpha_max = np.max(np.abs(gradient0))

and choose for the regularization strength :\math:`\alpha` a fraction of this critical value, e.g. ``alpha = 0.01 * alpha_max``.

Problem setup
=============

Consider the optimization problem:

.. math::
    \min_x f(x) + \lambda || x||_1

where:

- :math:`f: \mathbb{R}^d \to \mathbb{R}` is a convex differentiable function,
- :math:`|| x ||_1` is the L1 norm of :math:`x`,
- :math:`\lambda > 0` is the regularization parameter.

We aim to determine the conditions under which the solution to this problem is :math:`x = 0`.

Theoretical background
======================


Let

.. math::

    g(x) = f(x) + \lambda || x||_1

According to Fermat's rule, 0 is the minimizer of :math:`g` if and only if 0 is in the subdifferential of :math:`g` at 0.
The subdifferential of :math:`|| x ||_1` at 0 is the L-infinity unit ball:

.. math::
    \partial || \cdot ||_1 (0) = \{ u \in \mathbb{R}^d : ||u||_{\infty} \leq 1 \}

Thus,

.. math::
    :nowrap:

    \begin{equation}
        \begin{aligned}
        0 \in \text{argmin} ~ g(x)
        &\Leftrightarrow 0 \in \partial g(0) \\
        &\Leftrightarrow
        0 \in \nabla f(0) + \lambda \partial || \cdot ||_1 (0) \\
        &\Leftrightarrow - \nabla f(0)  \in  \lambda \{ u \in \mathbb{R}^d : ||u||_{\infty} \leq 1 \} \\
        &\Leftrightarrow || \nabla f(0) ||_\infty \leq \lambda
        \end{aligned}
    \end{equation}


We have just shown that the minimizer of :math:`g = f + \lambda || \cdot ||_1` is 0 if and only if :math:`\lambda \geq ||\nabla f(0)||_{\infty}`.

Example
=======

Consider the loss function for Ordinary Least Squares :math:`f(x) = \frac{1}{2n} ||Ax - b||_2^2`, where :math:`n` is the number of samples. We have:

.. math::
    \nabla f(x) = \frac{1}{n}A^T (Ax - b)

At :math:`x=0`:

.. math::
    \nabla f(0) = -\frac{1}{n}A^T b

The infinity norm of the gradient at 0 is:

.. math::
    ||\nabla f(0)||_{\infty} = \frac{1}{n}||A^T b||_{\infty}

For :math:`\lambda \geq \frac{1}{n}||A^T b||_{\infty}`, the solution to :math:`\min_x \frac{1}{2n} ||Ax - b||_2^2 + \lambda || x||_1` is :math:`x=0`.



References
==========

Refer to Section 3.1 and Proposition 4 in particular of [1] for more details.

.. _1:

[1] Eugene Ndiaye, Olivier Fercoq, Alexandre Gramfort, and Joseph Salmon. 2017. Gap safe screening rules for sparsity enforcing penalties. J. Mach. Learn. Res. 18, 1 (January 2017), 4671–4703.
