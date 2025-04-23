
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
   :page-layout: full

skglm
======
.. container:: hero-container

    .. container:: hero-text

        .. rubric:: skglm
            :class: hero-title


        .. container:: hero-description

            .. raw:: html

              The <strong>fastest</strong> and <strong>most modular</strong> Python package for regularized <strong>Generalized Linear Models</strong> — designed for researchers and engineers who demand speed, structure, and <strong><span class="highlight-orange">scikit-learn</span></strong> compatibility.

        .. container:: hero-buttons

            `Get Started <getting_started.html>`_

    .. container:: hero-gallery

        .. image:: _static/images/landingpage/hero.png
            :alt: Illustration showing sparse modeling and GLMs
            :class: hero-gallery-img
            :target: auto_examples/index.html


.. container:: section-spacer

    .. container:: section-intro

      .. rubric:: Simple. Modular. Powerful.
          :class: section-title

    .. container:: section-subtitle

        Everything you need to build fast, flexible, and scalable GLMs — in one modular library.

.. container:: features-grid

    .. container:: feature-box

        .. image:: _static/images/landingpage/ease.png
            :alt: Ease icon
            :class: feature-icon

        .. container:: feature-text

          .. rubric:: Easy to Use
              :class: feature-title

          Get started in minutes with an intuitive API, comprehensive examples, and out-of-the-box estimators.

    .. container:: feature-box

        .. image:: _static/images/landingpage/modular.png
            :alt: Modular icon
            :class: feature-icon

        .. container:: feature-text

            .. rubric:: Modular Design
                :class: feature-title

            Compose custom estimators from interchangeable datafits and penalties tailored to your use case.

    .. container:: feature-box

        .. image:: _static/images/landingpage/performance.png
            :alt: Performance icon
            :class: feature-icon

        .. container:: feature-text

            .. rubric:: Speed
                :class: feature-title

            Solve large-scale problems with lightning-fast solvers — up to 100× faster than ``scikit-learn``.

    .. container:: feature-box

        .. image:: _static/images/landingpage/compatible.png
            :alt: Compatibility icon
            :class: feature-icon

        .. container:: feature-text

            .. rubric:: Plug & Extend
                :class: feature-title

            Fully scikit-learn compatible and ready for custom research and production workflows.

.. container:: section-spacer

    .. container:: section-intro

      .. rubric:: Support Us
          :class: section-title

    .. container:: support-box

      .. rubric:: Citation
          :class: support-title
      Using ``skglm`` in your work? You are free to use it. It is licensed under
      `BSD 3-Clause <https://github.com/scikit-learn-contrib/skglm/blob/main/LICENSE>`_.
      As the result of perseverant academic research, the best way to support its development is by citing it.
      ::
            @inproceedings{skglm,
                title     = {Beyond L1: Faster and better sparse models with skglm},
                author    = {Q. Bertrand and Q. Klopfenstein and P.-A. Bannier
                             and G. Gidel and M. Massias},
                booktitle = {NeurIPS},
                year      = {2022},
            }

            @article{moufad2023skglm,
                title  = {skglm: improving scikit-learn for regularized Generalized Linear Models},
                author = {Moufad, Badr and Bannier, Pierre-Antoine and Bertrand, Quentin
                          and Klopfenstein, Quentin and Massias, Mathurin},
                year   = {2023}
            }

    .. container:: support-box

      .. rubric:: Contributions
          :class: support-title
      Contributions, improvements, and bug reports are always welcome. Help us make ``skglm`` better!

      .. container:: hero-buttons

          `How to Contribute <contribute.html>`_

.. container:: section-spacer

    .. container:: section-intro

      .. rubric:: Real-World Applications
          :class: section-title

    .. container:: section-subtitle

      ``skglm`` drives impactful solutions across diverse sectors with its fast, modular approach to regularized GLMs and sparse modeling.
      Find various advanced topics in our `Tutorials <tutorials/tutorials.html>`_ and `Examples <auto_examples/index.html>`_ sections.

    .. container:: applications-grid

        .. container:: application-box

            .. image:: _static/images/landingpage/healthcare.png
                :alt: Healthcare icon
                :class: application-icon

            .. container:: application-text

              .. rubric:: Healthcare
                  :class: application-title

              Enhance clinical trial analytics and early biomarker discovery by efficiently analyzing high-dimensional biological data and features like cox regression modeling.

        .. container:: application-box

            .. image:: _static/images/landingpage/finance.png
                :alt: Finance icon
                :class: application-icon

            .. container:: application-text

              .. rubric:: Finance
                  :class: application-title

              Conduct transparent and interpretable risk modeling with scalable, robust sparse regression across vast datasets.

        .. container:: application-box

            .. image:: _static/images/landingpage/energy.png
                :alt: Energy icon
                :class: application-icon

            .. container:: application-text

              .. rubric:: Energy
                  :class: application-title

              Optimize real-time electricity forecasting and load analysis by processing large time-series datasets for predictive maintenance and anomaly detection.

.. container:: sponsor-banner

  .. container:: sponsor-inline

    This project is made possible thanks to the support of

    .. image:: _static/images/landingpage/inrialogo.png
          :alt: Inria logo
          :class: sponsor-logo
          :target: https://www.inria.fr/en


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
