.. _maths_cox_datafit:

=============================
Mathematic behind Cox datafit
=============================

This tutorial presents the mathematics behind Cox datafit using both Breslow and Efron estimate. 


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

To get the expression of the Cox datafit, we refer to the expression of the negative log-likelihood according to Breslow estimate [`1`_, Section 2]

.. math::
    :label: breslow-estimate

    l(\beta) = \frac{1}{n} \sum_{i=1}^{n} -s_i \langle x_i, \beta \rangle + s_i \log(\sum_{y_j \geq y_i} e^{\langle x_j, \beta \rangle})
    .


Ideally, we ought to have a vectorized expression to ease gradient and Hessian derivation.

.. note::

    Having a vectorized expression would enable us to leverage ``Numpy`` broadcasting and vectorization
    and gain in terms of efficiency.


We introduce the matrix :math:`mathbf{B} \in \mathbb{R}^{n \times n}` defined as :math:`\mathbf{B}_{i, j} = \mathbb{1}_{y_j \geq y_i} = 1` if :math:`y_j \geq y_i` and :math:`0` otherwise.

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
    :label: vectorized-cox-breslow

    l(\beta) =  -\frac{1}{n} \langle s, \mathbf{X}\beta \rangle + \frac{1}{n} \langle s, \log(\mathbf{B}e^{\mathbf{X}\beta}) \rangle
    .

We observe from this vectorized reformulation that Cox datafit depends only :math:`\mathbf{X}\beta`. On the one hand, this illustrate that it fits well the GLM framework. On the other, now on, we can focus the simplified function

.. math::
    :label: simple-function

    F(u) = -\frac{1}{n} \langle s, u \rangle + \frac{1}{n} \langle s, \log(\mathbf{B}e^u) \rangle
    ,

to derive the gradient and Hessian.


Gradient and Hessian
--------------------

Now that we have a vectorized expression of the datafit, we can apply the chain rule to derive its gradient and Hessian.

For some :math:`u \in \mathbb{R}^n`, The gradient is

.. math::
    :label: raw-gradient

    \nabla F(u) = -\frac{1}{n} s + \frac{1}{n} [\text{diag}(e^u)\mathbf{B}^\top] \frac{s}{\mathbf{B}e^u}
    ,

and the Hessian reads

.. math::
    :label: raw-hessian

    \nabla^2 F(u) = \frac{1}{n} \text{diag}(e^u) \text{diag}(\mathbf{B}^\top \frac{s}{\mathbf{B}e^u}) - \frac{1}{n} \text{diag}(e^u) \mathbf{B}^\top \text{diag}(\frac{s}{(\mathbf{B}e^u)^2})\mathbf{B}\text{diag}(e^u)
    ,

where the fraction and the square operations are performed element-wise.

The Hessian, as it is, is costly to evaluate because of the right hand-side term. In particular, the latter involves a :math:`\mathcal{O}(n^3)` operations. We overcome this limitation by deriving a diagonal upper bound on the Hessian.

We construct such an upper bound by noticing that

#. the function :math:`F` is convex and hence :math:`\nabla^2 F(u)` is positive semi-definite
#. the second term is positive semi-definite.

Therefore, we have,

.. math::
    :label: diagonal-upper-bound

    \nabla^2 F(u) \leq \frac{1}{n} \text{diag}(e^u) \text{diag}(\mathbf{B}^\top \frac{s}{\mathbf{B}e^u})
    ,

where the inequality applies on the eigenvalues.

.. note::

    Having a diagonal Hessian would reduce the cost of evaluating the Hessian to :math:`\mathcal{O}(n)` instead of :math:`\mathcal{O}(n^3)`.
    A byproduct of that is also reducing the cost of evaluating matrix-vector operations involving the Hessian to :math:`\mathcal{O}(n)` instead
    of :math:`\mathcal{O}(n^2)`.



Efron estimate
==============

Datafit expression
------------------

Efron estimate refines Breslow by handling tied observations, *observations with identical occurrences' time*.
We can define :math:`y_{i_1}, \ldots, y_{i_m}` the unique times, assumed to be in total :math:`m` and

.. math::
    :label: def-H
    
    H_{y_{i_l}} = \{ i \ | \ s_i = 1 \ ;\ y_i = y_{i_l} \}
    ,
    
the set of uncensored observations with the same time :math:`y_{i_l}`.

Again, we refer to the expression of the negative log-likelihood according to Efron estimate [`2`_,  Section 6, equation (6.7)] to get the datafit formula

.. math::
    :label: efron-estimate

    l(\beta) = \frac{1}{n} \sum_{l=1}^{m} (
        \sum_{i \in H_{i_l}} - \langle x_i, \beta \rangle 
        + \sum_{i \in H_{i_l}} \log(\sum_{y_j \geq y_{i_l}} e^{\langle x_j, \beta \rangle} - \frac{\#(i) - 1}{ |H_{i_l} |}\sum_{j \in H_{i_l}} e^{\langle x_j, \beta \rangle}))
    ,

where :math:`| H_{i_l} |` stands for the cardinal of :math:`H_{i_l}`, and :math:`\#(i)` the index of observation :math:`i` in :math:`H_{i_l}`.

Ideally, we would like to rewrite this expression like  `<vectorized-cox-breslow>`_ to leverage the established results about the gradient and Hessian. A closer look reveals what distinguishes both expressions is the presence of a double sum and a second term in the :math:`\log`.

First, we can observe that :math:`\cup_{l=1}^{m} H_{i_l} = \{ i \ | \ s_i = 1 \}`, which enables fusing the two sums, for instance

.. math::

    \sum_{l=1}^{m}\sum_{i \in H_{i_l}} - \langle x_i, \beta \rangle = \sum_{i: s_i = 1} - \langle x_i, \beta \rangle = \sum_{i=1}^n -s_i \langle x_i, \beta \rangle = -\langle s, \mathbf{X}\beta \rangle
    .

On the other hand, the minus term within :math:`\log` can be rewritten as a linear term in :math:`mathbf{X}\beta`

.. math::

    - \frac{\#(i) - 1}{| H_{i_l} |}\sum_{j \in H_{i_l}} e^{\langle x_j, \beta \rangle} 
        = \sum_{j=1}^{n} -\frac{\#(i) - 1}{| H_{i_l} |} \ \mathbb{1}_{j \in H_{i_l}} \ e^{\langle x_j, \beta \rangle}
        = \sum_{j=1}^n a_{i,j} e^{\langle x_j, \beta \rangle}
        = \langle a_i, e^{\mathbf{X}\beta} \rangle
        ,

where :math:`a_i` is a vector in :math:`\mathbb{R}^n` chosen accordingly to preform the linear operation.

By defining the matrix :math:`\mathbf{A}` with rows :math:`(a_i)_{i \in [n]}`, we deduce the final expression

.. math::
    :label: vectorized-cox-efron

    l(\beta) =  -\frac{1}{n} \langle s, \mathbf{X}\beta \rangle +\frac{1}{n} \langle s, \log(\mathbf{B}e^{\mathbf{X}\beta} - \mathbf{A}e^{\mathbf{X}\beta}) \rangle
    .

Algorithm 1 provides an efficient procedure to evaluate :math:`\mathbf{A}v` for some :math:`v` in :math:`\mathbb{R}^n`.

.. image:: /_static/images/cox-tutorial/A_dot_v.png
    :width: 400
    :align: center
    :alt: Algorithm 1 to evaluate A dot v


Gradient and Hessian
--------------------

Now that we casted the Efron estimate in form similar to `<vectorized-cox-breslow>`_, the evaluation of gradient and the diagonal upper of the Hessian reduces to to subtracting a linear term. Algorithm  2 provides an efficient procedure to evaluate :math:`\mathbf{A}^\top v` for some :math:`v` in :math:`\mathbb{R}^n`.

.. image:: /_static/images/cox-tutorial/A_transpose_dot_v.png
    :width: 400
    :align: center
    :alt: Algorithm 1 to evaluate A transpose dot v

.. note::

    We notice that the complexity of both algorithms is :math:`\mathcal{O}(n)` despite intervening a matrix multiplication.
    This is due to the special structure of :math:`\mathbf{A}` which in the case of sorted observations has a block diagonal structure
    with each block having equal columns.

    Here is an illustration with sorted observations having group sizes of identical occurrences times :math:`3, 2, 1, 3` respectively

    .. image:: /_static/images/cox-tutorial/structure_matrix_A.png
        :width: 300
        :align: center
        :alt: Illustration of the structure of A when observations are sorted


Reference
=========

.. _1:
[1] DY Lin. On the Breslow estimator. Lifetime data analysis, 13:471–480, 2007.

.. _2:
[2] Bradley Efron. The efficiency of cox’s likelihood function for censored data. Journal of the
American statistical Association, 72(359):557–565, 1977.
