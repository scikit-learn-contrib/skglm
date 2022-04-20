skglm
========

``skglm`` is a library that provide better sparse generalized linear model for scikit-learn.
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
        title={Beyond L1: Faster and Better Sparse Models with skglm},
        author={Q. Bertrand and Q. Klopfenstein and P.-A. Bannier and G. Gidel and M. Massias},
        year={2022},
        url={https://arxiv.org/abs/2204.07826}
    }



Installing the development version
----------------------------------
First clone the repository available at https://github.com/mathurinm/skglm::

    $ git clone https://github.com/mathurinm/skglm.git
    $ cd skglm/

Then, install the package with::

    $ pip install -e .

To check if everything worked fine, you can do::

    $ python -c 'import skglm'

and it should not give any error message.