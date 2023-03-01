## Setup

Let $n, n_s \in \mathbb{N}$, with $n_s < n$,

Given $H_k \in \mathbb{R}^{n \times n_s}$ and $S_k \in \mathbb{R}^{n_s \times n_s}$, for $k \in [K]$, solve

$$
\min_{S \in \mathbb{R}^{n \times n}} \frac{1}{2} \lVert \underbrace{\sum_{k=1}^{K} S_k - H_k^\top S H_k}_{R} \rVert_F^2 + \lambda \lVert S \rVert_1
\enspace .
$$

Denote the datafit term: $$f(S) = \frac{1}{2} \lVert \sum_{k=1}^{K} S_k - H_k^\top S H_k\rVert_F^2 = \frac{1}{2} \lVert R \rVert_F^2$$

##  PGD algorithm

- Gradient: $$\nabla f(S) = -\sum_{k=1}^{K} H_k R H_k^\top$$
- Lipschitz constant:
    * expression of lipchitz constant
    $$L = \lVert \sum_{1 \leq i,j \leq K} (H_j^\top H_i)^2\rVert_2$$
    * For $R_1=R(S_1), R_2(S_2) \in \mathbb{R}^{n_s \times n_s}$,
    $$\lVert \sum_{k=1}^{K} H_k R_1 H_k^\top - \sum_{k=1}^{K} H_k R_2 H_k^\top \rVert_F^2 \leq \lVert \sum_{i,j} (H_j^\top H_i)^2\rVert_2 \lVert R_1 - R_2 \rVert_F^2$$
    * Doing the same for $\lVert R_1 - R_2 \rVert_F^2$
    $$\lVert R_1 - R_2 \rVert_F^2 \leq  \lVert \sum_{i,j} (H_j^\top H_i)^2\rVert_2 \lVert S_1 - S_2 \rVert_F^2$$
    * Hence for the gradients,
    $$\lVert \nabla F(S_1) - \nabla F(S_2) \rVert_F \leq \lVert \sum_{i,j} (H_j^\top H_i)^2\rVert_2 \lVert \lVert S_1 - S_2\rVert_F$$
