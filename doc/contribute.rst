.. _contribute:

Contribute to ``skglm``
=======================
-----------------------


``skglm`` is a continuous endeavour that relies on the community efforts to last and evolve.
Your contribution is welcome and highly valuable. It can be

**bug report**
    ``skglm`` runs continuously unit tests on the code base to prevent bugs.
    Help us tighten these tests by reporting any bug that you encountered while using ``skglm``.
    To do so, use the `issue section <https://github.com/scikit-learn-contrib/skglm/issues>`_ ``skglm`` repository.

**feature request**
    We are constantly improving ``skglm`` and we would like to align that with the user needs.
    We highly appreciate any suggestion to extend or add new features to ``skglm``.
    You can use the `the issue section <https://github.com/scikit-learn-contrib/skglm/issues>`_ to make suggestions.

**pull request**
    you may have fixed a bug, added a feature, or even fixed a small typo in the documentation, ... 
    You can submit a `pull request <https://github.com/scikit-learn-contrib/skglm/pulls>`_
    to integrate your changes and we will reach out to you shortly.

.. note::

    - If are willing to contribute with code to ``skglm``, check the section below to learn how to install
    the development version of ``skglm``



Setup ``skglm`` on your local machine
---------------------------------------

Here are key steps to help you setup ``skglm`` on your local machine in case you wanted to
contribute with code or documentation.

1. Fork the repository and run the following command to clone it on your local machine

.. code-block:: shell

    $ git clone https://github.com/{YOUR_GITHUB_USERNAME}/skglm.git


2. ``cd`` to ``skglm`` directory and install it in edit mode by running

.. code-block:: shell

    $ cd skglm
    $ pip install -e .


3. To run the gallery of examples and build the documentation, run

.. code-block:: shell

    $ cd doc
    $ pip install -r doc-requirements.txt
    $ make html
