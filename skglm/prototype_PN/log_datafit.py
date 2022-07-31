import numpy as np


class Pr_LogisticRegression:

    def __init__(self):
        pass

    def raw_gradient(self, y, Xw):
        """"""
        exp_yXw = np.exp(-y * Xw)
        return -y * exp_yXw / (1 + exp_yXw)

    def raw_hessian(self, y, Xw):
        """"""
        exp_yXw = np.exp(-y * Xw)
        return exp_yXw / (1 + exp_yXw) ** 2
