"""Solve Lasso problem using FISTA GPU-implementation.

Problem reads::

    min_w (1/2n) * ||y - Xw||^2 + lmbd * ||w||_1
"""
