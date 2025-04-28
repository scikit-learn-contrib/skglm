.. _tutorials:

.. meta::
   :description: Step-by-step skglm tutorials covering custom datafits, penalties, intercept computations, Cox datafit mathematics, group Lasso details, and regularization strategies.

=========
Tutorials
=========

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: How to Add a Custom Datafit
      :link: add_datafit.html
      :text-align: left

      Learn to add a custom datafit through a hands-on examples: Implementing a Poisson datafit.

   .. grid-item-card:: How to Add a Custom Penalty
      :link: add_penalty.html
      :text-align: left

      Learn to add a custom penalty by implementing the :math:`\ell_1` penalty.

   .. grid-item-card:: Computation of the Intercept
      :link: intercept.html
      :text-align: left

      Explore how ``skglm`` fits an unpenalized intercept.

   .. grid-item-card:: Mathematics behind Cox Datafit
      :link: cox_datafit.html
      :text-align: left

      Understand the mathematical foundation of Cox datafit and its applications in survival analysis.

   .. grid-item-card:: Details on the Group Lasso
      :link: prox_nn_group_lasso.html
      :text-align: left

      Mathematical details about the group Lasso, in particular with nonnegativity constraints.

   .. grid-item-card:: Understanding `alpha_max`
      :link: alpha_max.html
      :text-align: left

      Learn how to choose the regularization strength in :math:`\ell_1`-regularization?

.. toctree::
   :hidden:

   add_datafit
   add_penalty
   intercept
   cox_datafit
   prox_nn_group_lasso
   alpha_max
