<section align="center">

# ``skglm``

## A fast :zap: and modular :hammer_and_pick: scikit-learn replacement for sparse GLMs

</section>

![build](https://github.com/scikit-learn-contrib/skglm/workflows/pytest/badge.svg)
![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)
![Downloads](https://pepy.tech/badge/skglm/month)
![PyPI version](https://badge.fury.io/py/skglm.svg)


``skglm`` is a Python package that offers **fast estimators** for sparse Generalized Linear Models (GLMs) that are **100% compatible with ``scikit-learn``**. It is **highly flexible** and supports a wide range of GLMs. You get to choose from our already made estimators or **customize your own** by combing our available datafits and penalty.

Excited to have a tour on ``skglm`` [documentation](https://contrib.scikit-learn.org/skglm/) :memo:?

# Why ``skglm``?

``skglm`` is specifically conceived to solve sparse GLMs.
It supports many missing models in ``scikit-learn`` and ensures high performance.
There are several reasons to opt for ``skglm`` among which:

|  |  |
| --- | -------------- |
| **Speed** :zap: | Fast solvers able to tackle large datasets, either dense or sparse, with millions of features **up to 100 times faster** than ``scikit-learn``|
| **Modularity** :hammer_and_pick: | User-friendly API than enables **composing custom estimators** with any combination of its existing datafits and penalties |
| **Extensibility** :arrow_up_down: | Flexible design that makes it **simple and easy to implement new datafits and penalties**, a matter of few lines of code
| **Compatibility** :electric_plug: | Estimators **fully compatible with the ``scikit-learn`` API** and drop-in replacements of its GLM estimators
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

## Trying ``skglm``

Once you installed ``skglm``, you can run the following code snippet to fit an MCP Regression model on a toy dataset

```python
# import model to fit
from skglm.estimators import MCPRegression
# import util to create a toy dataset
from skglm.utils import make_correlated_data

# generate a top dataset
X, y, _ = make_correlated_data(n_samples=10, n_features=100)

estimator = MCPRegression()
estimator.fit(X, y)

# print R²
print(estimator.score(X, y))
```
You can refer to the documentation to explore the list of ``skglm``'s already-made estimators. 

Didn't find one that suits you :monocle_face:, you still can compose your own.
Here is a code snippet that fits the previous estimator with a Huber loss instead.

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

- **bug report**: you may encounter a bug while using ``skglm``. Don't hesitate to report it on the issue section.
- **feature request**: you may want to extend/add new features to ``skglm``. You can use the issue section to make suggestions.
- **pull request**: you may have fixed a bug, added a features, or even fixed a small typo in the documentation, ... you can submit a pull request and we will reach out to you asap.


# Cite

``skglm`` is the result of perseverant research. It is licensed under BSD 3-Clause.
You are free to use it and if you do so, please cite
 
```bibtex
@inproceedings{skglm,
    title   = {Beyond L1 norm with skglm},
    author  = {Q. Bertrand and Q. Klopfenstein and P.-A. Bannier and G. Gidel and M. Massias},
    journal = {arXiv preprint arXiv:2204.07826},
    url     = {https://arxiv.org/pdf/2204.07826.pdf}
    year    = {2022},
}
```


# Useful links

- link to documentation: https://contrib.scikit-learn.org/skglm/
- link to ``skglm`` arXiv article: https://arxiv.org/pdf/2204.07826.pdf
