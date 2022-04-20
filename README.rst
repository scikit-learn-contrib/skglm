=====

|image0| |image1|


``skglm`` is a library that provide better sparse generalized linear model for scikit-learn.
Its main features are:

- **speed**: problems with millions of features can be solved in seconds. Default solvers rely on efficient coordinate descent with numba just in time compilation.
- **flexibility**: virtually any combination of datafit and penalty can be implemented in a few lines of code.
- **sklearn API**: all estimators are drop-in replacements for scikit-learn.
- **scope**: support for many missing models in scikit-learn - weighted Lasso, arbitrary group penalties, non convex sparse penalties, etc.


Currently, the package handles any combination of the following datafits:

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

First clone the repository available at https://github.com/mathurinm/skglm::

    $ git clone https://github.com/mathurinm/skglm.git
    $ cd skglm/

Then, install the package with::

    $ pip install -e .

To check if everything worked fine, you can do::

    $ python -c 'import skglm'

and it should not give any error message.



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
----

If you use this code, please cite

.. code-block:: none

    @online{skglm,
        title={Beyond L1: Faster and Better Sparse Models with skglm},
        author={Q. Bertrand and Q. Klopfenstein and P.-A. Bannier and G. Gidel and M. Massias},
        year={2022},
        url={https://arxiv.org/abs/2204.07826}




ArXiv links:

- https://arxiv.org/pdf/2204.07826.pdf

.. |image0| image:: https://github.com/mathurinm/skglm/workflows/build/badge.svg
   :target: https://github.com/mathurinm/skglm/actions?query=workflow%3Abuild
.. |image1| image:: https://codecov.io/gh/mathurinm/skglm/branch/main/graphs/badge.svg?branch=main
   :target: https://codecov.io/gh/mathurinm/skglm
