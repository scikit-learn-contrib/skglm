# Computation of the intercept in $\texttt{skglm}$

This short document intends to give insights and guidance for the computation of the intercept coefficient within the $\texttt{skglm}$ solvers.

Let the desgin matrix be $X\in \mathbb{R}^{n\times p}$ where $n$ is the number of samples and $p$ the number of features. 
We denote $\beta\in\mathbb{R}^p$ the coefficients of the Generalized Linear Model and $\beta_0$ its intercept. 
In the way $\texttt{skglm}$ is designed, computing the intercept coefficients does not only imply adding an extra column of ones in the design matrix as the intercept $\beta_0$ is unpenalized, meaning that only the datafitting term $F$ depends on the intercept. The given optimization problem can then be written as:

$$
\begin{align}
    \beta^\star, \beta_0^\star
    \in
    \argmin_{\beta \in \mathbb{R}^p, \beta_0 \in \mathbb{R}}
    \Phi(\beta)
    \triangleq
    \underbrace{F(X\beta + \beta_0\boldsymbol{1}_{n})}_{\triangleq f(\beta)}
    + \sum_{j=1}^p g_j(\beta_j)
    \enspace ,
\end{align}
$$
where $\boldsymbol{1}_{n}$ is the vector of size $n$ composed with only ones.


The solvers of $\texttt{skglm}$ update the intercept after each epoch of coordinate descent by doing a gradient descent update.
$$
\begin{align}
    \beta^{(k+1)}_0 = \beta^{(k)}_0 - \frac{1}{L_0}\nabla_{\beta_0}F(X\beta^{(k)} + \beta_0^{(k)}\boldsymbol{1}_{n}) 
    \enspace ,
\end{align}
$$
where $L_0$ is the lispchitz constant associated to the intercept.


The convergence criterion computed for the gradient is then only the absolute value of the gradient with respect to $\beta_0$ since the intercept optimality condition, for a solution $\beta^\star$, $\beta_0^\star$ is just:
$$
\begin{align}
    \nabla_{\beta_0}F(X\beta^\star + \beta_0^\star\boldsymbol{1}_{n}) = 0
    \enspace .
\end{align}
$$
Moreover, we have that 
$$
\begin{align}
    \nabla_{\beta_0}F(X\beta^\star + \beta_0^\star\boldsymbol{1}_{n}) = \boldsymbol{1}_{n}^\top \nabla F(X\beta^\star + \beta_0^\star\boldsymbol{1}_{n})
    \enspace .
\end{align}
$$

We will now derive the update used in Equation 2 for three different datafitting functions. 

---

## The Quadratic datafit

We define 
$$
\begin{align}
    F(X\beta + \beta_0\boldsymbol{1}_{n})) = \frac{1}{2n} \lVert y - X\beta + \beta_0\boldsymbol{1}_{n} \rVert^2_2
    \enspace .
\end{align}
$$
In this case $\nabla F(z) = \frac{1}{n}(z - y)$ hence Eq. 4 is equal to:
$$
\begin{align}
    \nabla_{\beta_0}F(X\beta + \beta_0\boldsymbol{1}_{n}) = \frac{1}{n}\sum_{i=1}^n(X_{i:}\beta + \beta_0 - y_i)
    \enspace .
\end{align}
$$
Finally, the Lispchitz constant $L_0 = \frac{1}{n}\sum_{i=1}^n 1^2 = 1$.

The quantity $\frac{1}{L_0}\nabla_{\beta_0}F(X\beta^{(k)} + \beta_0^{(k)}\boldsymbol{1}_{n})$ is called $\texttt{intercept\_update\_step}$ in the class $\texttt{Quadratic}$ of $\texttt{skglm}$.

---

## The Logistic datafit

In this case, 
$$
\begin{align}
    F(X\beta + \beta_0\boldsymbol{1}_{n}) = \frac{1}{n} \sum_{i=1}^n \log(1 + \exp(-y_i X_{i:}\beta))
\end{align}
$$

We can then write
$$
\begin{align}
 \nabla_{\beta_0}F(X\beta + \beta_0\boldsymbol{1}_{n}) = \frac{1}{n} \sum_{i=1}^n  \frac{-y_i}{1 + \exp(- y_iX_{i:}\beta)} \enspace .
\end{align}
$$

Finally, the Lispchitz constant $L_0 = \frac{1}{4n}\sum_{i=1}^n 1^2 = \frac{1}{4}$.

---

## The Huber datafit

In this case, 
$$
\begin{align}
    F(X\beta + \beta_0\boldsymbol{1}_{n}) = \frac{1}{n} \sum_{i=1}^n f_{\delta}(y_i - X_{i:}\beta) \enspace ,
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

Let $r_i = y_i - X_{i:}\beta$ We can then write
$$
\begin{align}
 \nabla_{\beta_0}F(X\beta + \beta_0\boldsymbol{1}_{n}) = \frac{1}{n} \sum_{i=1}^n r_i\mathbb{1}_{\{|y_i - X_{i:}\beta|\leq\delta\}} + \text{sign}(r_i)\delta\mathbb{1}_{\{|y_i - X_{i:}\beta|>\delta\}} \enspace ,
\end{align}
$$
where $1_{x > \delta}$ is the classical indicator function.

Finally, the Lispchitz constant $L_0 = \frac{1}{n}\sum_{i=1}^n 1^2 = 1$.
