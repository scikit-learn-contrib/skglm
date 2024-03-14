.. _api_documentation:

=================
API Documentation
=================

.. currentmodule:: skglm

Estimators
==========

.. currentmodule:: skglm

.. autosummary::
   :toctree: generated/

   GeneralizedLinearEstimator
   CoxEstimator
   ElasticNet
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
   WeightedGroupL2_Plus_L1
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
   Quadratic
   QuadraticGroup
   QuadraticSVC


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
   SqrtQuadratic
   SqrtLasso