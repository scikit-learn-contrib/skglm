.. skglm documentation master file, created by
   sphinx-quickstart on Mon May 23 16:22:52 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

skglm
========

skglm is a library that provide better sparse generalized linear model for scikit-learn.
Its main features are:

- **speed**: problems with millions of features can be solved in seconds. Default solvers rely on efficient coordinate descent with numba just in time compilation.
- **flexibility**: virtually any combination of datafit and penalty can be implemented in a few lines of code.
- **sklearn API**: all estimators are drop-in replacements for scikit-learn.
- **scope**: support for many missing models in scikit-learn - weighted Lasso, arbitrary group penalties, non convex sparse penalties, etc.


Cite
----

If you use this code, please cite

.. code-block:: none

    @online{skglm,
        title={Beyond L1 norm with skglm},
        author={Q. Bertrand and Q. Klopfenstein and P.-A. Bannier and G. Gidel and M. Massias},
        journal = {arXiv preprint arXiv:2204.07826},
        url={https://arxiv.org/pdf/2204.07826.pdf}
        year={2022},
    }


Installing the development version
----------------------------------
First clone the repository available at https://github.com/scikit-learn-contrib/skglm::

    $ git clone https://github.com/scikit-learn-contrib/skglm.git
    $ cd skglm/

Then, install the package with::

    $ pip install -e .

To check if everything worked fine, you can do::

    $ python -c 'import skglm'

and it should not give any error message.


API
---

.. toctree::
    :maxdepth: 1

    api.rst
    whats_new.rst
