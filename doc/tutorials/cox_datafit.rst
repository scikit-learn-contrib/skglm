.. _maths_cox_datafit:

=============================
Mathematic behind Cox datafit
=============================

This tutorial presents the mathematics behind Cox datafit using both estimate Breslow and Efron.


Problem setup
=============

Let's consider a typical survival analysis setup with

- :math:`\mathbf{X} \in \mathbb{R}^{n \times p}` is a matrix of :math:`p` predictors and :math:`n` samples :math:`x_i \in \mathbb{R}^p`
- :math:`y \in \mathbb{R}^n` a vector recording the time of events occurrences
- :math:`s \in \{ 0, 1 \}^n` a binary vector (observation censorship) where :math:`1` means *event occurred*

where we are interested in estimating the vector of coefficient :math:`\beta \in \mathbb{R}^p`.



Breslow estimate
================

Datafit expression
------------------

To get the expression of the Cox datafit, we refer to the expression of the negative log-likelihood according to Breslow estimate [1, Section 2]

.. math::
    :label: breslow-estimate

    l(\beta) = \sum_{i=1}^{n} -s_i \langle x_i, \beta \rangle + s_i \log(\sum_{y_j \geq y_i} e^{\langle x_j, \beta \rangle})
    .


Ideally, we ought to have a more compact expression to ease gradient and Hessian derivation as well as leverage vectorization.
For that we introduce the matrix :math:`mathbf{B} \in \mathbb{R}^{n \times n}` defined as :math:`\mathbf{B}_{i, j} = \mathbb{1}_{y_j \geq y_i} = 1, \text{ if } y_j \geq y_i \text{ and } 0 \text{ otherwise}`.

We notice that the first term in the sum can we rewritten as

.. math::

    \sum_{i=1}^{n} -s_i \langle x_i, \beta \rangle = -\langle s, \mathbf{X}\beta \rangle
    ,

whereas the second term

.. math::

    \sum_{i=1}^n s_i \log(\sum_{y_j \geq y_i} e^{\langle x_j, \beta \rangle}) = \sum_{i=1}^n s_i \log(\sum_{j=1}^n \mathbb{1}_{y_j \geq y_i} e^{\langle x_j, \beta \rangle}) = \langle s, \log(\mathbf{B}e^{\mathbf{X}\beta}) \rangle
    ,

where the :math:`\log` is performed element-wise. Therefore we deduce the expression of Cox datafit

.. math::
    :label: cox-datafit

    l(\beta) =  -\langle s, \mathbf{X}\beta \rangle + \langle s, \log(\mathbf{B}e^{\mathbf{X}\beta}) \rangle
    .

We observe from this vectorized reformulation that Cox datafit depends only :math:`\mathbf{X}\beta`. On the one hand, this illustrate that it fits well the GLM framework. On the other, now on, we can focus the simplified function

.. math::

    F(u) = -\langle s, u \rangle + \langle s, \log(\mathbf{B}e^u) \rangle
    ,

to derive the gradient and Hessian.


Gradient and Hessian
--------------------

Now that we have a vectorized expression of the datafit, we can apply the chain rule to derive its gradient and Hessian.

For some :math:`u \in \mathbb{R}^n`, The gradient is

.. math::
    :label: raw-gradient

    \nabla F(u) = -s + [\text{diag}(e^u)\mathbf{B}^\top] \frac{s}{\mathbf{B}e^u}
    ,

and the Hessian reads

.. math::
    :label: raw-hessian

    \nabla^2 F(u) = \text{diag}(e^u) \text{diag}(\mathbf{B}^\top \frac{s}{\mathbf{B}e^u}) - \text{diag}(e^u) \mathbf{B}^\top \text{diag}(\frac{s}{(\mathbf{B}e^u)^2})\mathbf{B}\text{diag}(e^u)
    ,

where the fraction and the square operations are performed element-wise.

The Hessian, as it is, is costly to evaluate because of the right hand-side term. In particular, the latter involves a :math:`\mathcal{O}(n^3)` operation. We overcome this limitation by deriving a diagonal upper bound on the Hessian.

We construct such an upper bound by noticing that: 1) the :math:`F` is convex and hence :math:`\nabla^2 F(u)` is positive semi-definite, and 2) the second terms is also positive semi-definite. Therefore, we have,

.. math::
    :label: diagonal-upper-bound

    \nabla^2 F(u) \leq \text{diag}(e^u) \text{diag}(\mathbf{B}^\top \frac{s}{\mathbf{B}e^u})
    ,

where the inequality apply on the eigenvalues.


Reference
=========

[1] DY Lin. On the breslow estimator. Lifetime data analysis, 13:471–480, 2007.

[2] Bradley Efron. The efficiency of cox’s likelihood function for censored data. Journal of the
American statistical Association, 72(359):557–565, 1977.
