from .base import BaseDatafit, BaseMultitaskDatafit  # noqa F401

from .single_task import (  # noqa F401
    Quadratic, QuadraticSVC, Logistic,
    Huber,
)

from .multi_task import QuadraticMultiTask  # noqa F401

from .group import QuadraticGroup  # noqa F401
