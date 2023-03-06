## Content

1. [Setup](#setup)
1. [PGD algorithm](#cd-algorithm)
    * [Formulation](#formulation)
    * [Cost](#algorithm-cost)
1. [CD algorithm](#pgd-algorithm)
    * [Formulation](#details)
    * [Cost](#algorithm-cost-1)


# Setup

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

Denotes by $h^{k}_i \in \mathbb{R}^{n_s}$ the $i$-th row of $H_k$ 



# CD algorithm

## Formulation

- reformulation:
$$\min_{s_{i,j}} \sum_{k=1}^{K} \frac{1}{2} \lVert S_k - \sum_{i,j} s_{i,j} h^{k}_i h^{k}_j {}^\top \rVert_F^2 + \lambda \sum_{i,j} \lvert s_{i,j} \rvert$$

can be derived using the [connection between matrix product and outer product](https://math.stackexchange.com/questions/2335457/matrix-at-a-as-sum-of-outer-products)

- Derivative w.r.t. $i,j$ coordinate
$$\partial_{i, j} f(S) = \sum_{k=1}^{K} \langle h^{k}_i \otimes {h^{k}_j}, R_k\rangle_F = \sum_{k=1}^{K} \langle h^{k}_i , R_k {h^{k}_j} \rangle$$
- second derivative w.r.t. $i,j$ coordinate
$$\partial_{i, j}^2 f(S) = \sum_{k=1}^{K} \lVert h^{k}_i \otimes {h^{k}_j}\lVert_F^2 = \sum_{k=1}^{K} (\lVert h^{k}_i \rVert \lVert h^{k}_j \rVert)^2$$

## Algorithm cost
Cost per epoch is $\mathcal{O}(K \times n^2 \times n_s)$.
Detail of the cost per one pass
1. compute grad at $(i, j)$, cost $\mathcal{O}(K \times n_s^2)$
$$
\mathrm{grad}_{i, j} \gets \sum_{k=1}^{K} \langle h^{k}_i , R_k {h^{k}_j} \rangle
$$
2. update $s_{i, j}$, cost $\mathcal{O}(1)$
$$
s_{i,j} \gets \mathrm{ST}_{\frac{\lambda}{L_{i,j}}} (s_{i,j} - \frac{1}{L_{i,j}} \mathrm{grad}_{i, j})
$$
3. update residuals, $\mathcal{O}(K \times n_s^2)$
$$
\begin{aligned}
\mathrm{For \ k=1 \ldots K}:\\
R_k & \gets R_k + \delta s_{i,j} \ h^{k}_i \otimes {h^{k}_j} \\  
\end{aligned}
$$

#  PGD algorithm

## Details

- Gradient: $$\nabla f(S) = \sum_{k=1}^{K} -H_k R_k(S) H_k^\top$$
- Lipschitz constant:
    * expression of lipchitz constant
    $$L \leq \sum_{k=1}^K  \lVert H_k \rVert_2^4$$
    * The Lipchitz constant of a function is upper bounded by the sum of lipchitz constants of its sum terms
    * For each term, namely for $k \in [K]$, and $S, S' \in \mathbb{R}^{n \times n}$,
    $$\lVert H_k R_k(S) H_k^\top - H_k R_k(S') H_k^\top \rVert_F^2 \leq \lVert (H_k^\top H_k)^2 \rVert_2 \lVert S - S' \rVert_F^2$$
    * upper bound on the spectral norm
    $$\lVert (H_k^\top H_k)^2 \rVert_2 \leq \lVert H_k \rVert_2^4$$

## Algorithm cost
Cost per iteration is $\mathcal{O}(K \times n^2 \times n_s)$.
Details:
1. Compute grad $\mathcal{O}(K \times n^2 \times n_s)$
$$
\begin{aligned}
\mathrm{For \ k=1 \ldots K}:\\
R_k & \gets H_k^\top S H_k \\  
\mathrm{grad} &\gets \mathrm{grad} - H_k R_k H_k^\top  \\
\end{aligned}
$$
2. Prox grad update update $\mathcal{O}(n^2)$
$$
S \gets \mathrm{ST}_{\frac{\lambda}{L}}(S - \frac{1}{L} \mathrm{grad})
$$