.. _api:

.. meta::
   :description: Browse the skglm API documentation covering estimators (Lasso, ElasticNet, Cox), penalties (L1, SCAD, MCP), datafits (Logistic, Poisson), and optimized solvers.

=================
API
=================

.. currentmodule:: skglm

Estimators
==========

.. currentmodule:: skglm

.. autosummary::
   :toctree: generated/

   GeneralizedLinearEstimator
   GeneralizedLinearEstimatorCV
   CoxEstimator
   ElasticNet
   GroupLasso
   Lasso
   LinearSVC
   SparseLogisticRegression
   MCPRegression
   MultiTaskLasso
   WeightedLasso


Penalties
=========


.. currentmodule:: skglm.penalties

.. autosummary::
   :toctree: generated/

   IndicatorBox
   L0_5
   L1
   L1_plus_L2
   L2
   L2_3
   LogSumPenalty
   MCPenalty
   PositiveConstraint
   WeightedL1
   WeightedGroupL2
   WeightedMCPenalty
   SCAD
   BlockSCAD
   SLOPE


Datafits
========

.. currentmodule:: skglm.datafits

.. autosummary::
   :toctree: generated/

   Cox
   Gamma
   Huber
   Logistic
   LogisticGroup
   Poisson
   PoissonGroup
   Quadratic
   QuadraticGroup
   QuadraticHessian
   QuadraticSVC
   WeightedQuadratic


Solvers
=======

.. currentmodule:: skglm.solvers

.. autosummary::
   :toctree: generated/

   AndersonCD
   FISTA
   GramCD
   GroupBCD
   GroupProxNewton
   LBFGS
   MultiTaskBCD
   ProxNewton


Experimental
============

.. currentmodule:: skglm.experimental

.. autosummary::
   :toctree: generated/

   IterativeReweightedL1
   PDCD_WS
   Pinball
   QuantileHuber
   SmoothQuantileRegressor
   SqrtQuadratic
   SqrtLasso
