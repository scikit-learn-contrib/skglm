"""
===============================
Gamma and Poisson for modelling
===============================
"""

# Author: Pierre-Antoine Bannier<pierreantoine.bannier@gmail.com>

import numpy as np
from skglm.datafits import Gamma, Poisson
from skglm.penalties import L1



# Poisson model is typically used to model counts (non-negative integer values)
