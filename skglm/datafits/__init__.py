from .base import BaseDatafit, BaseMultitaskDatafit
from .single_task import (Quadratic, QuadraticSVC, Logistic, Huber, Poisson, Gamma,
                          Cox, WeightedQuadratic, QuadraticHessian,)
from ._double_quadratic import DoubleQuadratic
from .multi_task import QuadraticMultiTask
from .group import QuadraticGroup, LogisticGroup, PoissonGroup


__all__ = [
    BaseDatafit, BaseMultitaskDatafit,
    Quadratic, QuadraticSVC, Logistic, Huber, Poisson, Gamma, Cox,
    QuadraticMultiTask,
    QuadraticGroup, LogisticGroup, PoissonGroup, WeightedQuadratic,
    QuadraticHessian, DoubleQuadratic  # Add this
]