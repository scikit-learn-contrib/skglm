## Setup

Let $n, n_s \in \mathbb{N}$, with $n_s < n$,

Given $H_k \in \mathbb{R}^{n \times n_s}$ and $S_k \in \mathbb{R}^{n_s \times n_s}$, for $k \in [K]$, solve

$$
\min_{S \in \mathbb{R}^{n \times n}} \frac{1}{2} \lVert \underbrace{\sum_{k=1}^{K} S_k - H_k^\top S H_k}_{R} \rVert^2 + \lambda \lVert S \rVert_1
\enspace .
$$

Denote the datafit term: $$f(S) = \frac{1}{2} \lVert \sum_{k=1}^{K} S_k - H_k^\top S H_k\rVert^2 = \frac{1}{2} \lVert R \rVert^2$$

##  PGD algorithm

- Gradient: $$\nabla f(S) = \sum_{i=1}^{K} H_k R H_k^\top$$
- Lipschitz constant ?

