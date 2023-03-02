## Setup

Let $n, n_s \in \mathbb{N}$, with $n_s < n$,

Given $H_k \in \mathbb{R}^{n \times n_s}$ and $S_k \in \mathbb{R}^{n_s \times n_s}$, for $k \in [K]$, solve

$$
\min_{S \in \mathbb{R}^{n \times n}} \sum_{k=1}^{K} \frac{1}{2} \lVert  \underbrace{S_k - H_k^\top S H_k}_{R_k(S)} \rVert_F^2 + \lambda \lVert S \rVert_1
\enspace .
$$

Denote the datafit term:
$$
f(S) = \sum_{k=1}^K \frac{1}{2} \lVert R_k(S) \rVert_F^2
\enspace .
$$

##  PGD algorithm

- Gradient: $$\nabla f(S) = \sum_{k=1}^{K} -H_k R_k(S) H_k^\top$$
- Lipschitz constant:
    * expression of lipchitz constant
    $$L \leq \sum_{k=1}^K  \lVert H_k \rVert_2^4$$
    * The Lipchitz constant of a function is upper bounded by the sum of lipchitz constants of its sum terms
    * For each term, namely for $k \in [K]$, and $S, S' \in \mathbb{R}^{n \times n}$,
    $$\lVert H_k R_k(S) H_k^\top - H_k R_k(S') H_k^\top \rVert_F^2 \leq \lVert (H_k^\top H_k)^2 \rVert_2 \lVert S - S' \rVert_F^2$$
    * upper bound on the spectral norm
    $$\lVert (H_k^\top H_k)^2 \rVert_2 \leq \lVert H_k \rVert_2^4$$
