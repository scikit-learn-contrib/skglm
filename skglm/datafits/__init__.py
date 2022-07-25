from .base import BaseDatafit, BaseMultitaskDatafit  # noqa F401

from .single_task import (  # noqa F401
    Quadratic, QuadraticSVC, QuadraticSVC_32, Logistic, Logistic_32,
    Huber, Huber_32,
)

from .multi_task import QuadraticMultiTask  # noqa F401

from .group import QuadraticGroup  # noqa F401
