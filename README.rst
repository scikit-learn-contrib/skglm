skglm
=====

|image0| |image1|

Fast algorithm in a modular package to solve Lasso (and beyond)-like problems with coordinate descent + working sets + Anderson acskglmation, under a scikit-learn API.
The solvers used allow for solving large scale problems with millions of features, up to 100 times faster than scikit-learn.

Currently, the package handles any combination of the following datafit:

- quadratic
- binary cross entropy
- multitask quadratic

and the following penalties:

- L1 norm
- MCP
- SCAD
- L05 and L2/3 penalties


The estimators follow the scikit-learn API, come with automated parallel cross-validation, and support both sparse and dense data.

.. with optionally feature centering, normalization, and unpenalized intercept fitting.

Documentation
=============

Please visit https://mathurinm.github.io/skglm/ for the latest version
of the documentation.

Install the released version
============================

Assuming you have a working Python environment, e.g., with Anaconda you
can `install skglm with pip <https://pypi.python.org/pypi/skglm/>`__.

From a console or terminal install skglm with pip:

::

    pip install -U skglm

Install and work with the development version
=============================================

From a console or terminal clone the repository and install skglm:

::

    git clone https://github.com/mathurinm/skglm.git
    cd skglm/
    pip install -e .

To build the documentation you will need to run:


::

    pip install -U sphinx_gallery sphinx_bootstrap_theme
    cd doc/
    make html


Demos & Examples
================

In the `example section <https://mathurinm.github.io/skglm/auto_examples/index.html>`__ of the documentation,
you will find numerous examples on real life datasets,
timing comparison with other estimators, easy and fast ways to perform cross validation, etc.


Dependencies
============

All dependencies are in the ``./requirements.txt`` file.
They are installed automatically when ``pip install -e .`` is run.

Cite
====

If you use this code, please cite:

::

    @article{skglm,
      title = 	 {Beyond L1: Faster and Better Sparse Models with skglm},
      author = 	 {Quentin Bertrand and Quentin Klopfenstein and Pierre-Antoine Bannier and Gauthier Gidel and Mathurin Massias},
      year = 	 {2022},
    }



ArXiv links:

- https://arxiv.org/pdf/2204.07826.pdf

.. |image0| image:: https://github.com/mathurinm/skglm/workflows/build/badge.svg
   :target: https://github.com/mathurinm/skglm/actions?query=workflow%3Abuild
.. |image1| image:: https://codecov.io/gh/mathurinm/skglm/branch/main/graphs/badge.svg?branch=main
   :target: https://codecov.io/gh/mathurinm/skglm
