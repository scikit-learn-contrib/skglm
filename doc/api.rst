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
   L2_3
   MCPenalty
   WeightedL1
   WeightedGroupL2
   SCAD
   BlockSCAD
   SLOPE


Datafits
========

.. currentmodule:: skglm.datafits

.. autosummary::
   :toctree: generated/

   Huber
   Logistic
   LogisticGroup
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
   MultiTaskBCD
   ProxNewton


Experimental
============

.. currentmodule:: skglm.experimental

.. autosummary::
   :toctree: generated/

   SqrtLasso