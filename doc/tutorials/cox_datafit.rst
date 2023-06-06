.. _maths_cox_datafit:

=============================
Mathematic behind Cox datafit
=============================

This tutorial presents the mathematics behind Cox datafit using both estimate Breslow and Efron.


Problem setup
=============

Let's consider a typical survival analysis setup with

- :math:`\mathbf{X} \in \mathbb{R}^{n \times p}` is a matrix of :math:`p` predictors and :math:`n` samples :math:`x_i \in \mathbb{R}^p`
- :math:`y \in \mathbb{R}^n` a vector recording the time of events occurrences
- :math:`s \in \{ 0, 1 \}^n` a binary vector (observation censorship) where :math:`1` means *event occurred*

where we are interested in estimating a the vector of coefficient :math:`\beta \in \mathbb{R}^p`.



