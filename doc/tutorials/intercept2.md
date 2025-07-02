This note gives insights and guidance for the handling of an intercept coefficient within the `skglm` solvers.

Let the design matrix be $X in RR^{n times p}$ where $n$ is the number of samples and $p$ the number of features.
We denote $beta in RR^p$ the coefficients of the Generalized Linear Model and $beta_0$ its intercept.
In many packages such as `liblinear`, the intercept is handled by adding an extra column of ones in the design matrix. This is costly in memory, and may lead to different solutions if all coefficients are penalized, as the intercept $beta_0$ is usually not.
`skglm` follows a different route and solves directly:

```{math}
    beta^star, beta_0^star
    in
    underset(beta in RR^p, beta_0 in RR)("argmin")
    Phi(beta)
    triangleq
    underbrace(F(X beta + beta_0 bb"1"_n))_(triangleq f(beta, beta_0))
    + sum_(j=1)^p g_j(beta_j)
    \ ,
```


where $bb"1"_{n}$ is the vector of size $n$ composed only of ones.


The solvers of `skglm` update the intercept after each update of $beta$ by doing a (1 dimensional) gradient descent update:

```{math}
    beta_0^((k+1)) = beta_0^((k)) - 1/(L_0) nabla_(beta_0)F(X beta^((k)) + beta_0^((k)) bb"1"_{n})
    \ ,
```

where $L_0$ is the Lipschitz constant associated to the intercept.
The local Lipschitz constant $L_0$ satisfies the following inequality

$$
\forall x, x_0 in RR^p times RR, \forall h in RR, |nabla_(x_0) f(x, x_0 + h) - nabla_(x_0) f(x, x_0)| <= L_0 |h| \ .
$$

This update rule should be implemented in the `intercept_update_step` method of the datafit class.

The convergence criterion computed for the gradient is then only the absolute value of the gradient with respect to $beta_0$ since the intercept optimality condition, for a solution $beta^star$, $beta_0^star$ is:

```{math}
    nabla_(beta_0)F(X beta^star + beta_0^star bb"1"_n) = 0
    \ ,
```

Moreover, we have that

```{math}
    nabla_(beta_0) F(X beta + beta_0 bb"1"_n) = bb"1"_n^\top nabla_beta F(X beta + beta_0 bb"1"_n)
    \ .
```


We will now derive the update used in Equation 2 for three different datafitting functions.

---

## The Quadratic datafit

We define

```{math}
    F(X beta + beta_0 bb"1"_n) = 1/(2n) norm(y - X beta -  beta_0 bb"1"_{n})_2^2
    \ .
```

In this case $nabla f(z) = 1/n (z - y)$ hence Eq. 4 is equal to:

```{math}
    nabla_(beta_0) F(X beta + beta_0 bb"1"_n) = 1/n sum_(i=1)^n (X_( i: ) beta + beta_0 - y_i)
    \ .
```

Finally, the Lipschitz constant is $L_0 = 1/n sum_(i=1)^n 1^2 = 1$.



---

## The Logistic datafit

In this case,

```{math}
    F(X beta + beta_0 bb"1"_{n}) = 1/n sum_(i=1)^n log(1 + exp(-y_i(X_( i: ) beta + beta_0 bb"1"_n))
```


We can then write

```{math}
    nabla_(beta_0) F(X beta + beta_0 bb"1"_n) = 1/n sum_(i=1)^n  (-y_i)/(1 + exp(-y_i(X_( i: ) beta + beta_0 bb"1"_n))) \ .
```


Finally, the Lipschitz constant is $L_0 = 1/(4n) sum_(i=1)^n 1^2 = 1/4$.

---

## The Huber datafit

In this case,

```{math}
    F(X beta + beta_0 bb"1"_{n}) = 1/n sum_(i=1)^n f_(delta) (y_i - X_( i: ) beta - beta_0 bb"1"_n) \ ,
```

where

```{math}
    f_delta(x) = {
        (1/2 x^2, if x <= delta),
        (delta |x| - 1/2 delta^2, if x > delta)
    :} \ .
```


Let $r_i = y_i - X_( i: ) beta - beta_0 bb"1"_n$. We can then write

```{math}
 nabla_(beta_0) F(X beta + beta_0 bb"1"_{n}) = 1/n sum_(i=1)^n r_i bbb"1"_({|r_i| <= delta}) + "sign"(r_i) delta bbb"1"_({|r_i| > delta}) \ ,
```

where $bbb"1"_({x > delta})$ is the classical indicator function.

Finally, the Lipschitz constant is $L_0 = 1/n sum_(i=1)^n 1^2 = 1$.
