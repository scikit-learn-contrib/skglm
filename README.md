<section align="center">

# ``skglm``

## A fast :zap: and modular :hammer_and_pick: scikit-learn replacement for sparse GLMs

</section>

![build](https://github.com/scikit-learn-contrib/skglm/workflows/pytest/badge.svg)
![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)
![Downloads](https://pepy.tech/badge/skglm/month)
![PyPI version](https://badge.fury.io/py/skglm.svg)


``skglm`` is a Python package that offers **fast estimators** for sparse Generalized Linear Models (GLMs) that are **100% compatible with ``scikit-learn``**. It is **highly flexible** and supports a wide range of GLMs. You get to choose from our already made estimators or **customize your own** by combing our available datafits and penalty.


# Why ``skglm``?

``skglm`` is conceived to support many GLMs problems while being performant. There are several reasons to opt for it among which

- **Speed**:
Fast solvers able to tackle large datasets, either dense or sparse, with millions of features **up to 100 times faster** than ``scikit-learn``
- **Modularity**:
User-friendly API than enables **composing custom estimators** with any combination of its existing datafits and penalties
- **Extensibility**: Flexible design that makes it **simple and easy to implement new datafits and penalties**, a matter of few lines of code.   
- **Compatibility**:
Estimators **fully compatible with the ``scikit-learn`` API** and drop-in replacements of its GLM estimators



# Getting started with ``skglm``

## Installing ``skglm``

```shell
pip install -U skglm
```

```shell
conda install skglm
```

## Trying ``skglm``



# Contribute to ``skglm``




# Cite

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

- link to documentation
- link to article







- sparse Generalized Linear Models
- flexible design, support a wide range of models

- introduce ``skglm`` 
- what is the problem we address
- why using ``skglm`` instead of scikit-learn
- what are the value brought by ``skglm``
- what we offer


some terms:
- datafit: ensure model data fidelity 
- penalty: enforce a sparse structure of the solution
- modularity: the ability to combine existing datafits and penalties
- extensibility: the ability to add support of new datafit and/or penalty


- Python package, 100% written in Python

- flexibility, high performance (up 100 times faster than scikit-learn)
- Sparse Generalized Linear Models
- Flexibility, myriad of models


- Large spectrum of problems: datafits, penalties
- Estimator compatible with scikit-learn API: ``.fit`` method, ``Pipelines``, ``GridSearchCV``
- high flexibility in adding new models


