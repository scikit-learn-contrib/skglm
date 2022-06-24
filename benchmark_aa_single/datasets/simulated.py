from benchopt import BaseDataset
from benchopt import safe_import_context

from benchopt.datasets.simulated import make_correlated_data

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):
    name = "Simulated"

    parameters = {
        'n_samples, n_features': [(300, 3000)],
    }

    def __init__(self, n_samples, n_features, random_state=0):
        self.n_samples, self.n_features = n_samples, n_features
        self.random_state = random_state

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        X, y, _ = make_correlated_data(self.n_samples, self.n_features,
                                       random_state=rng)

        return {'X': X, 'y': y}
