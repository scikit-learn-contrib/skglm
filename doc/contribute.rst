.. _contribute:

Contribute to ``skglm``
=======================

``skglm`` is a continuous endeavour that relies on community efforts to last and evolve.
Your contribution is welcome and highly valuable. You can help with

**bug report**
    ``skglm`` runs unit tests on the codebase to prevent bugs.
    Help us tighten these tests by reporting any bug that you encounter.
    To do so, use the `issue section <https://github.com/scikit-learn-contrib/skglm/issues>`_.

**feature request**
    We are constantly improving ``skglm`` and we would like to align that with the user needs.
    We highly appreciate any suggestion to extend or add new features to ``skglm``.
    You can use the `the issue section <https://github.com/scikit-learn-contrib/skglm/issues>`_ to make suggestions.

**pull request**
    You may have fixed a bug, added a feature, or even fixed a small typo in the documentation...
    You can submit a `pull request <https://github.com/scikit-learn-contrib/skglm/pulls>`_
    to integrate your changes and we will reach out to you shortly.
    If this is your first pull request, you can refer to `this scikit-learn guide <https://scikit-learn.org/stable/developers/contributing.html#how-to-contribute>`_.

As part of the `scikit-learn-contrib <https://github.com/scikit-learn-contrib>`_ GitHub organization, we adopt the scikit-learn `code of conduct <https://github.com/scikit-learn/scikit-learn/blob/main/CODE_OF_CONDUCT.md>`_.

.. note::

    If you are willing to contribute with code to ``skglm``, check the section below to learn how to install the development version.



Setup ``skglm`` on your local machine
---------------------------------------

Here are the key steps to help you setup ``skglm`` on your machine in case you want to
contribute with code or documentation.

1. `Fork the repository <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>`_ and run the following command to clone it on your local machine, make sure to replace ``{YOUR_GITHUB_USERNAME}`` with your GitHub username

.. code-block:: shell

    $ git clone https://github.com/{YOUR_GITHUB_USERNAME}/skglm


2. ``cd`` to ``skglm`` directory and install it in edit mode by running

.. code-block:: shell

    $ cd skglm
    $ pip install -e .


3. To build the documentation locally, run

.. tab-set::

    .. tab-item:: with plots in the example gallery

        .. code-block:: shell

            $ cd doc
            $ pip install .[doc]
            $ make html

    .. tab-item:: without plots in the example gallery

        .. code-block:: shell

            $ cd doc
            $ pip install .[doc]
            $ make html-noplot
