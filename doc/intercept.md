# Computation of the intercept in `skglm`

This note gives insights and guidance for the handling of an intercept coefficient within the $\texttt{skglm}$ solvers.

Let the design matrix be $X\in \mathbb{R}^{n\times p}$ where $n$ is the number of samples and $p$ the number of features.
We denote $\beta\in\mathbb{R}^p$ the coefficients of the Generalized Linear Model and $\beta_0$ its intercept.
In many packages such as `liblinear`, the intercept is handled by adding an extra column of ones in the design matrix. This is costly in memory, and may lead to different solutions if all coefficients are penalized, as the intercept $\beta_0$ is usually not.
`skglm` follows a different route and solves directly:

$$
\begin{align}
    \beta^\star, \beta_0^\star
    \in
    \argmin_{\beta \in \mathbb{R}^p, \beta_0 \in \mathbb{R}}
    \Phi(\beta)
    \triangleq
    \underbrace{F(X\beta + \beta_0\boldsymbol{1}_{n})}_{\triangleq f(\beta, \beta_0)}
    + \sum_{j=1}^p g_j(\beta_j)
    \enspace ,
\end{align}
$$
where $\boldsymbol{1}_{n}$ is the vector of size $n$ composed only of ones.


The solvers of $\texttt{skglm}$ update the intercept after each update of $\beta$ by doing a (1 dimensional) gradient descent update:
$$
\begin{align}
    \beta^{(k+1)}_0 = \beta^{(k)}_0 - \frac{1}{L_0}\nabla_{\beta_0}F(X\beta^{(k)} + \beta_0^{(k)}\boldsymbol{1}_{n})
    \enspace ,
\end{align}
$$
where $L_0$ is the Lipschitz constant associated to the intercept.
The local Lipschitz constant $L_0$ statisfies the following inequality
$$
\forall x, x_0\in \mathbb{R}^p \times \mathbb{R}, \forall h \in \mathbb{R}, |\nabla_{x_0} f(x, x_0 + h) - \nabla_{x_0} f(x, x_0)| \leq L_0 |h| \enspace .
$$
This update rule should be implemented in the $\texttt{intercept\_update\_step}$ method of the datafit class.

The convergence criterion computed for the gradient is then only the absolute value of the gradient with respect to $\beta_0$ since the intercept optimality condition, for a solution $\beta^\star$, $\beta_0^\star$ is:
$$
\begin{align}
    \nabla_{\beta_0}F(X\beta^\star + \beta_0^\star\boldsymbol{1}_{n}) = 0
    \enspace ,
\end{align}
$$
Moreover, we have that
$$
\begin{align}
    \nabla_{\beta_0}F(X\beta + \beta_0\boldsymbol{1}_{n}) = \boldsymbol{1}_{n}^\top \nabla_\beta F(X\beta + \beta_0\boldsymbol{1}_{n})
    \enspace .
\end{align}
$$

We will now derive the update used in Equation 2 for three different datafitting functions.

---

## The Quadratic datafit

We define
$$
\begin{align}
    F(X\beta + \beta_0\boldsymbol{1}_{n}) = \frac{1}{2n} \lVert y - X\beta - \beta_0\boldsymbol{1}_{n} \rVert^2_2
    \enspace .
\end{align}
$$
In this case $\nabla f(z) = \frac{1}{n}(z - y)$ hence Eq. 4 is equal to:
$$
\begin{align}
    \nabla_{\beta_0}F(X\beta + \beta_0\boldsymbol{1}_{n}) = \frac{1}{n}\sum_{i=1}^n(X_{i:}\beta + \beta_0 - y_i)
    \enspace .
\end{align}
$$
Finally, the Lipschitz constant is $L_0 = \frac{1}{n}\sum_{i=1}^n 1^2 = 1$.



---

## The Logistic datafit

In this case,
$$
\begin{align}
    F(X\beta + \beta_0\boldsymbol{1}_{n}) = \frac{1}{n} \sum_{i=1}^n \log(1 + \exp(-y_i(X_{i:}\beta + \beta_0\boldsymbol{1}_n))
\end{align}
$$

We can then write
$$
\begin{align}
 \nabla_{\beta_0}F(X\beta + \beta_0\boldsymbol{1}_{n}) = \frac{1}{n} \sum_{i=1}^n  \frac{-y_i}{1 + \exp(- y_i(X_{i:}\beta + \beta_0\boldsymbol{1}_n))} \enspace .
\end{align}
$$

Finally, the Lipschitz constant is $L_0 = \frac{1}{4n}\sum_{i=1}^n 1^2 = \frac{1}{4}$.

---

## The Huber datafit

In this case,
$$
\begin{align}
    F(X\beta + \beta_0\boldsymbol{1}_{n}) = \frac{1}{n} \sum_{i=1}^n f_{\delta}(y_i - X_{i:}\beta - \beta_0\boldsymbol{1}_n)) \enspace ,
\end{align}
$$
where
$$
\begin{align}
    f_\delta(x) = \begin{cases}
            \frac{1}{2}x^2 & \text{if } x \leq \delta \\
            \delta |x| - \frac{1}{2}\delta^2 & \text{if } x > \delta
           \end{cases} \enspace .
\end{align}
$$

Let $r_i = y_i - X_{i:}\beta - \beta_0\boldsymbol{1}_n$. We can then write
$$
\begin{align}
 \nabla_{\beta_0}F(X\beta + \beta_0\boldsymbol{1}_{n}) = \frac{1}{n} \sum_{i=1}^n r_i\mathbb{1}_{\{|r_i|\leq\delta\}} + \text{sign}(r_i)\delta\mathbb{1}_{\{|r_i|>\delta\}} \enspace ,
\end{align}
$$
where $1_{x > \delta}$ is the classical indicator function.

Finally, the Lipschitz constant is $L_0 = \frac{1}{n}\sum_{i=1}^n 1^2 = 1$.
