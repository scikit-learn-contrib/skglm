.. skglm documentation master file, created by
   sphinx-quickstart on Mon May 23 16:22:52 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. meta::
   :og:title: skglm: Fast, Scalable & Flexible Regularized GLMs and Sparse Modeling for Python
   :description: skglm is the fastest, most modular Python library for regularized GLMs—fully scikit-learn compatible for advanced statistical modeling.
   :og:image: _static/images/logo.svg
   :og:url: https://contrib.scikit-learn.org/skglm/
   :keywords: Generalized Linear Models, GLM, scikit-learn, Lasso, ElasticNet, Cox, modular, efficient, regularized

=========
``skglm``
=========
*— Fast and Flexible Generalized Linear Models for Python —*

--------


``skglm`` is a Python package that offers **fast estimators** for regularized Generalized Linear Models (GLMs)
that are **100% compatible with** ``scikit-learn``. It is **highly flexible** and supports a wide range of GLMs,
designed to tackle high-dimensional data and scalable machine learning problems.
Whether you choose from our ready-made estimators or **customize your own** using a modular combination of datafits and penalties,
skglm delivers performance and flexibility for both academic research and production environments.

Get a hands-on glimpse on ``skglm`` through the :ref:`Getting started page <getting_started>`.


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
    }

    @article{moufad2023skglm,
        title={skglm: improving scikit-learn for regularized Generalized Linear Models},
        author={Moufad, Badr and Bannier, Pierre-Antoine and Bertrand, Quentin and Klopfenstein, Quentin and Massias, Mathurin},
        year={2023}
    }


Why ``skglm``?
--------------

``skglm`` is specifically conceived to solve regularized GLMs.
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

Applications and Use Cases
---------------------------

``skglm`` drives impactful solutions across diverse sectors with its fast, modular approach to regularized GLMs and sparse modeling. Some examples include:

.. list-table::
    :widths: 20 80

    * - **Healthcare:**
      - Enhance clinical trial analytics and early biomarker discovery by efficiently analyzing high-dimensional biological data and features like cox regression modeling.
    * - **Finance:**
      - Conduct transparent and interpretable risk modeling with scalable, robust sparse regression across vast datasets.
    * - **Energy:**
      - Optimize real-time electricity forecasting and load analysis by processing large time-series datasets for predictive maintenance and anomaly detection.

Other advanced topics and uses-cases are covered in :ref:`Tutorials <tutorials>`.

.. it is mandatory to keep the toctree here although it doesn't show up in the page
.. when adding/modifying pages, don't forget to update the toctree

.. toctree::
    :maxdepth: 1
    :hidden:
    :includehidden:

    getting_started.rst
    tutorials/tutorials.rst
    auto_examples/index.rst
    api.rst
    contribute.rst
    changes/whats_new.rst
