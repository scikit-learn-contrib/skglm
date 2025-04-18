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


.. raw:: html

   <section class="hero-container" style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; padding: 2rem 0;">
     <div class="hero-text" style="flex: 1; min-width: 300px; padding: 1rem;">
       <h1 style="font-size: 3rem; color: var(--pst-color-primary); font-weight: 800; margin-bottom: 1rem;">skglm</h1>
       <p style="font-size: 1.25rem; line-height: 1.6; max-width: 40rem; color: #1c1c1c;">
          The <strong>fastest</strong> and <strong>most modular</strong> Python package for regularized <strong> Generalized Linear Models</strong> — designed for researchers and engineers who demand speed, structure, and <strong style="color: #e76f00;">scikit-learn</strong> compatibility.
      </p>
       <div class="hero-buttons">
         <a class="hero-button primary" href="getting_started.html">Get Started</a>
       </div>
     </div>
     <div class="hero-gallery" style="flex: 1; min-width: 300px; text-align: center; padding: 1rem;">
       <a href="auto_examples/index.html">
         <img src="_static/images/landingpage/hero.png" alt="Illustration showing sparse modeling and GLMs" class="hero-gallery-img" style="max-width: 100%; height: auto;">
       </a>
     </div>
    </section>

   <hr style="margin: 3rem 0;">

  <section style="margin-top: 2rem; text-align: center;">
    <h2 style="font-size: 2.5rem; color: #1c1c1c; font-family: inherit; font-weight: 600; margin-bottom: 0.25rem;">
      Simple. Modular. Powerful.
    </h2>
    <p style="font-size: 1.25rem; font-weight: 400; color: #4a4a4a; font-family: inherit; max-width: 52rem; margin: 0 auto 2.5rem auto;">
      Everything you need to build fast, flexible, and scalable GLMs — in one modular library.
    </p>
  </section>

   <section style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 2rem 3rem; align-items: start; justify-items: start;">
    <div style="background: #f9fafb; padding: 2rem; border-radius: 8px; height: 100%; display: flex; align-items: center; gap: 1.5rem;">
      <img src="_static/images/landingpage/ease.png" alt="Ease icon" style="width: 4.5rem; height: 4.5rem;">
      <div>
        <h3 style="font-size: 1.2rem; color: #1f77b4; margin: 0 0 0.5rem 0;">Easy to Use</h3>
        <p style="margin: 0;">Get started in minutes with intuitive an API, comprehensive examples, and out-of-the-box estimators.</p>
      </div>
    </div>
    <div style="background: #f9fafb; padding: 2rem; border-radius: 8px; height: 100%; display: flex; align-items: center; gap: 1.5rem;">
      <img src="_static/images/landingpage/modular.png" alt="Modular icon" style="width: 4.5rem; height: 4.5rem;">
      <div>
        <h3 style="font-size: 1.2rem; color: #1f77b4; margin: 0 0 0.5rem 0;">Modular Design</h3>
        <p style="margin: 0;">Compose custom estimators from interchangeable datafits and penalties tailored to your use case.</p>
      </div>
    </div>
    <div style="background: #f9fafb; padding: 2rem; border-radius: 8px; height: 100%; display: flex; align-items: center; gap: 1.5rem;">
      <img src="_static/images/landingpage/performance.png" alt="Performance icon" style="width: 4.5rem; height: 4.5rem;">
      <div>
        <h3 style="font-size: 1.2rem; color: #1f77b4; margin: 0 0 0.5rem 0;">Speed</h3>
        <p style="margin: 0;">Solve large-scale problems with lightning-fast solvers — up to 100× faster than <code style="background-color: #f5f5f5; padding: 0.1rem 0.3rem; border-radius: 4px;">scikit-learn</code>.</p>
      </div>
    </div>
    <div style="background: #f9fafb; padding: 2rem; border-radius: 8px; height: 100%; display: flex; align-items: center; gap: 1.5rem;">
      <img src="_static/images/landingpage/compatible.png" alt="Compatibility icon" style="width: 4.5rem; height: 4.5rem;">
      <div>
        <h3 style="font-size: 1.2rem; color: #1f77b4; margin: 0 0 0.5rem 0;">Plug & Extend</h3>
        <p style="margin: 0;">Fully scikit-learn compatible and ready for custom research and production workflows.</p>
      </div>
  </section>

    <section style="margin-top: 2rem; text-align: center;">
    <h2 style="font-size: 2.5rem; color: #1c1c1c; font-family: inherit; font-weight: 600; margin-bottom: 2 rem;">
      Support Us
    </h2>

          <div style="background: #f9fafb; padding: 2rem; border-radius: 8px; text-align: center;">
        <h3 style="font-size: 1.2rem; color: #1f77b4; margin: 0 0 0.5rem 0;">Citation</h3>
        <p style="margin-bottom: 1rem;">
          Using <code>skglm</code> in your work? You are free to use it. It is licensed under
          <a href="https://github.com/scikit-learn-contrib/skglm/blob/main/LICENSE" target="_blank">BSD 3-Clause</a>. As the result of perseverant academic research, the best way to support its development is by citing it.
        </p>
        <pre style="font-size: 0.85rem; background: #f4f4f4; padding: 1rem; border-radius: 6px; overflow-x: auto; text-align: left;">


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
        </pre>
      </div>

      <div style="background: #f9fafb; padding: 2rem; border-radius: 8px; text-align: center;">
        <h3 style="font-size: 1.2rem; color: #1f77b4; margin: 0 0 0.5rem 0;">Contributions</h3>
        <p style="margin-bottom: 1rem;">Contributions, improvements, and bug reports are always welcome. Help us make <code>skglm</code> better!</p>
        <div class="hero-buttons">
         <a class="hero-button primary" href="contribute.html">How to Contribute</a>
       </div>
      </div>
    </section>

  <section style="margin-top: 4rem; text-align: center;">
    <h2 style="font-size: 2.5rem; color: #1c1c1c; font-family: inherit; font-weight: 600; margin-bottom: 0.25rem;">
      Real-World Applications
    </h2>
    <p style="font-size: 1.1rem; font-weight: 400; color: #4a4a4a; font-family: inherit; max-width: 52rem; margin: 0 auto 2.5rem auto;">
      <code>skglm</code> drives impactful solutions across diverse sectors with its fast, modular approach to regularized GLMs and sparse modeling. Find various advanced topics and in our <a href="tutorials/tutorials.html">Tutorials</a> and <a href="auto_examples/index.html">Examples</a> sections.
    </p>

    <section style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem; text-align: center; max-width: 1000px; margin: 0 auto;">
      <div style="background: #f9fafb; padding: 1.5rem; border-radius: 8px;">
        <img src="_static/images/landingpage/healthcare.png" alt="Healthcare icon" style="width: 5rem; height: 5rem">
        <h3 style="font-size: 1.1rem; font-weight: 600; color: #1c1c1c; margin-bottom: 0.5rem;">Healthcare</h3>
        <p style="margin: 0;">Enhance clinical trial analytics and early biomarker discovery by efficiently analyzing high-dimensional biological data and features like cox regression modeling.</p>
      </div>
      <div style="background: #f9fafb; padding: 1.5rem; border-radius: 8px;">
        <img src="_static/images/landingpage/finance.png" alt="Finance icon" style="width: 5rem; height: 5rem">
        <h3 style="font-size: 1.1rem; font-weight: 600; color: #1c1c1c; margin-bottom: 0.5rem;">Finance</h3>
        <p style="margin: 0;">Conduct transparent and interpretable risk modeling with scalable, robust sparse regression across vast datasets.</p>
      </div>
      <div style="background: #f9fafb; padding: 1.5rem; border-radius: 8px;">
        <img src="_static/images/landingpage/energy.png" alt="Energy icon" style="width: 5rem; height: 5rem">
        <h3 style="font-size: 1.1rem; font-weight: 600; color: #1c1c1c; margin-bottom: 0.5rem;">Energy</h3>
        <p style="margin: 0;">Optimize real-time electricity forecasting and load analysis by processing large time-series datasets for predictive maintenance and anomaly detection.</p>
      </div>
    </section>
  </section>

  <section style="margin-top: 4rem; padding: 1.5rem 2rem; background-color: #f9fafb; display: flex; align-items: center; justify-content: center; gap: 1rem; flex-wrap: wrap; border-radius: 8px;">
    <p style="font-size: 1.1rem; color: #1c1c1c; font-family: inherit; margin: 0;">
      This project is made possible thanks to the support of
      </p>
    <a href="https://www.inria.fr/en" target="_blank" rel="noopener">
      <img src="_static/images/landingpage/inrialogo.png" alt="Inria logo" style="height: 30px;">
    </a>
  </section>

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
