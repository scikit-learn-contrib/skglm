<section align="center">

<img src="doc/_static/images/logo.svg" alt="skglm logo" width="34%" align="center"/>

## A fast ⚡ and modular ⚒️ scikit-learn replacement for sparse GLMs

![build](https://github.com/scikit-learn-contrib/skglm/workflows/pytest/badge.svg)
![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)
[![Downloads](https://static.pepy.tech/badge/skglm)](https://pepy.tech/project/skglm)
[![Downloads](https://static.pepy.tech/badge/skglm/month)](https://pepy.tech/project/skglm)
[![PyPI version](https://badge.fury.io/py/skglm.svg)](https://pypi.org/project/skglm/)


</section>


``skglm`` is a Python package that offers **fast estimators** for sparse Generalized Linear Models (GLMs) that are **100% compatible with ``scikit-learn``**. It is **highly flexible** and supports a wide range of GLMs.
You get to choose from ``skglm``'s already-made estimators or **customize your own** by combining the available datafits and penalties.

Excited to have a tour on ``skglm`` [documentation](https://contrib.scikit-learn.org/skglm/)?


# Why ``skglm``?

``skglm`` is specifically conceived to solve sparse GLMs.
It supports many missing models in ``scikit-learn`` and ensures high performance.
There are several reasons to opt for ``skglm`` among which:

|  |  |
| ----- | -------------- |
| **Speed** | Fast solvers able to tackle large datasets, either dense or sparse, with millions of features **up to 100 times faster** than ``scikit-learn``|
| **Modularity** | User-friendly API that enables **composing custom estimators** with any combination of its existing datafits and penalties |
| **Extensibility** | Flexible design that makes it **simple and easy to implement new datafits and penalties**, a matter of few lines of code
| **Compatibility** | Estimators **fully compatible with the ``scikit-learn`` API** and drop-in replacements of its GLM estimators
|  |  |


# Get started with ``skglm``

## Installing ``skglm``

``skglm`` is available on PyPi. Run the following command to get the latest version of the package

```shell
pip install -U skglm
```

It is also available on Conda _(not yet, but very soon...)_ and can be installed via the command

```shell
conda install skglm
```

## First steps with ``skglm``

Once you installed ``skglm``, you can run the following code snippet to fit a MCP Regression model on a toy dataset

```python
# import model to fit
from skglm.estimators import MCPRegression
# import util to create a toy dataset
from skglm.utils import make_correlated_data

# generate a toy dataset
X, y, _ = make_correlated_data(n_samples=10, n_features=100)

# init and fit estimator
estimator = MCPRegression()
estimator.fit(X, y)

# print R²
print(estimator.score(X, y))
```
You can refer to the documentation to explore the list of ``skglm``'s already-made estimators.

Didn't find one that suits you? you can still compose your own.
Here is a code snippet that fits a MCP-regularized problem with Huber loss.

```python
# import datafit, penalty and GLM estimator
from skglm.datafits import Huber
from skglm.penalties import MCPenalty
from skglm.estimators import GeneralizedLinearEstimator

from skglm.utils import make_correlated_data

X, y, _ = make_correlated_data(n_samples=10, n_features=100)
# create and fit GLM estimator with Huber loss and MCP penalty
estimator = GeneralizedLinearEstimator(
    datafit=Huber(delta=1.),
    penalty=MCPenalty(alpha=1e-2, gamma=3),
)
estimator.fit(X, y)
```

You will find detailed description on the supported datafits and penalties and how to combine them in the API section of the documentation.
You can also take our tutorial to learn how to create your own datafit and penalty.


# Contribute to ``skglm``

``skglm`` is a continuous endeavour that relies on the community efforts to last and evolve. Your contribution is welcome and highly valuable. It can be

- **bug report**: you may encounter a bug while using ``skglm``. Don't hesitate to report it on the [issue section](https://github.com/scikit-learn-contrib/skglm/issues).
- **feature request**: you may want to extend/add new features to ``skglm``. You can use [the issue section](https://github.com/scikit-learn-contrib/skglm/issues) to make suggestions.
- **pull request**: you may have fixed a bug, added a features, or even fixed a small typo in the documentation, ... you can submit a [pull request](https://github.com/scikit-learn-contrib/skglm/pulls) and we will reach out to you asap.


# Cite

``skglm`` is the result of perseverant research. It is licensed under [BSD 3-Clause](https://github.com/scikit-learn-contrib/skglm/blob/main/LICENSE). You are free to use it and if you do so, please cite

```bibtex
@inproceedings{skglm,
    title     = {Beyond L1: Faster and better sparse models with skglm},
    author    = {Q. Bertrand and Q. Klopfenstein and P.-A. Bannier and G. Gidel and M. Massias},
    booktitle = {NeurIPS},
    year      = {2022},
}
```


# Useful links

- link to documentation: https://contrib.scikit-learn.org/skglm/
- link to ``skglm`` arXiv article: https://arxiv.org/pdf/2204.07826.pdf
