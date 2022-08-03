import numpy as np
from skglm.utils import sigmoid


class Pr_LogisticRegression:

    def __init__(self):
        pass

    def get_spec(self):
        spec = ()
        return spec

    def params_to_dict(self):
        return dict()

    def value(self, y, w, Xw):
        return np.log(1. + np.exp(- y * Xw)).sum() / len(y)

    def gradient_scalar(self, X, y, w, Xw, j):
        return - X[:, j] @ (y * sigmoid(- y * Xw)) / len(y)

    def raw_gradient(self, y, Xw):
        """"""
        return -y * sigmoid(-y * Xw) / len(y)

    def raw_hessian(self, y, Xw, grad):
        """"""
        return -grad * (y + len(y) * grad)
