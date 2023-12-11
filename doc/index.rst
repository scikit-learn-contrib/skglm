.. skglm documentation master file, created by
   sphinx-quickstart on Mon May 23 16:22:52 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=========
``skglm``
=========
*— A fast and modular scikit-learn replacement for sparse GLMs —*

--------


``skglm`` is a Python package that offers **fast estimators** for sparse Generalized Linear Models (GLMs)
that are **100% compatible with** ``scikit-learn``. It is **highly flexible** and supports a wide range of GLMs.
You get to choose from ``skglm``'s already-made estimators or **customize your own** by combining the available datafits and penalties.

Get a hands-on glimpse on ``skglm`` through the :ref:`Getting started page <getting_started>`.


Why ``skglm``?
--------------

``skglm`` is specifically conceived to solve sparse GLMs.
It supports many missing models in ``scikit-learn`` and ensures high performance.

There are several reasons to opt for ``skglm`` among which:

.. list-table::
    :widths: 20 80

    * - **Speed**
      - Fast solvers able to tackle large datasets, either dense or sparse, with millions of features **up to 100 times faster** than ``scikit-learn``
    * - **Modularity**
      - User-friendly API that enables **composing custom estimators** with any combination of its existing datafits and penalties
    * - **Extensibility**
      - Flexible design that makes it **simple and easy to implement new datafits and penalties**, a matter of few lines of code
    * - **Compatibility**
      - Estimators **fully compatible with the** ``scikit-learn`` **API** and drop-in replacements of its GLM estimators


Installing ``skglm``
--------------------

``skglm`` is available on PyPi. Get the latest version of the package by running

.. code-block:: shell

    $ pip install -U skglm

It is also available on conda-forge and can be installed using, for instance:

.. code-block:: shell

    $ conda install -c conda-forge skglm

With ``skglm`` being installed, Get the first steps with the package via the :ref:`Getting started section <getting_started>`.
Other advanced topics and uses-cases are covered in :ref:`Tutorials <tutorials>`.

.. note::

  - Currently, ``skglm`` is unavailable on Conda but will be released very soon...


Cite
----

``skglm`` is the result of perseverant research. It is licensed under
`BSD 3-Clause <https://github.com/scikit-learn-contrib/skglm/blob/main/LICENSE>`_.
You are free to use it and if you do so, please cite

.. code-block:: bibtex

    @inproceedings{skglm,
        title     = {Beyond L1: Faster and better sparse models with skglm},
        author    = {Q. Bertrand and Q. Klopfenstein and P.-A. Bannier
                     and G. Gidel and M. Massias},
        booktitle = {NeurIPS},
        year      = {2022},


Explore the documentation
-------------------------

.. toctree::
    :maxdepth: 1

    getting_started.rst
    tutorials/tutorials.rst
    auto_examples/index.rst
    api.rst
    contribute.rst
    changes/whats_new.rst

