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

Denotes the $i$-th row of $H_k$ by $h^{(k)}_i \in \mathbb{R}^{n_s}$

## CD algorithm

- reformulation:
$$\min_{s_{i,j}} \sum_{k=1}^{K} \frac{1}{2} \lVert S_k - \sum_{i,j} s_{i,j} h^{(k)}_i h^{(k)}_j {}^\top \rVert_F^2 + \lambda \sum_{i,j} \lvert s_{i,j} \rvert$$

can be derived using the [connection between matrix product and outer product of the rows/columns](https://math.stackexchange.com/questions/2335457/matrix-at-a-as-sum-of-outer-products)

- Derivative w.r.t. $i,j$ coordinate
$$\partial_{i, j} f(S) = \sum_{k=1}^{K} \langle h^{(k)}_i {h^{(k)}_j}^\top, R_k\rangle_F$$
- second derivative w.r.t. $i,j$ coordinate
$$\partial_{i, j}^2 f(S) = \sum_{k=1}^{K} \lVert h^{(k)}_i {h^{(k)}_j}^\top\lVert_F^2$$


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
