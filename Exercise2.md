We will search for a bifurcation caused by a change in stability of the fixed point at the origin, which can be explained by the eigenvalues of the Jacobian at the origin. Specifically, this translates to investigating for which $g$ there exists an eigenvalue with positive real part [[1]](https://eng.libretexts.org/Bookshelves/Industrial_and_Systems_Engineering/Chemical_Process_Dynamics_and_Controls_(Woolf)/10%3A_Dynamical_Systems_Analysis/10.04%3A_Using_eigenvalues_and_eigenvectors_to_find_stability_and_solve_ODEs).

First, we will rewrite the leading differential equation. The transfer function $\phi(x)=tanh(x)$ can be approximated around $x=0$ by $\phi(x)\approx x$.

The leading differential equation then becomes

$$\tau \frac{dr_i}{dt}=−r_i+\sum_{j=1}^{N}J_{ij}r_j$$

Since $(J\vec{r})_i=\sum_{j=1}^{N} J_{ij}r_{j}$, this equation can be generalized for all neurons to be

$$\tau \frac{d\vec{r}}{dt}=−\vec{r}+J\vec{r}=(J-I)\vec{r}$$

Secondly, to calculate the Jacobian of this system, we will evaluate the $ij$-th entries of the Jacobian individually.

$$\text{Jacobian}_{ij}=\frac{\partial}{\partial r_j}\left( -r_i +\sum_{k=1}^N J_{ik}r_k \right)$$

As the only term in $\sum_{k=1}^N J_{ik}r_k$ that depends on $r_j$ is the term where $k=j$, we have

$$\text{Jacobian}_{ij}=-\delta_{ij} +\frac{\partial}{\partial r_j}\left( J_{ij}r_j \right) = J_{ij}-\delta_{ij}$$

One can notice that the entries of this Jacobian are equivalent to those of $J-I$, so

$$\text{Jacobian}=J-I$$

Because each entry in $J$ is sampled from a normal distribution centered at 0, its spectrum follows the circular law with radius $R=\sigma \sqrt{N}=\frac{g}{\sqrt{N}}\sqrt{N}=g$ (in the thermodynamic limit, where $N \rightarrow \infty $) [[2]](https://en.wikipedia.org/wiki/Circular_law). Note also that the entries on the diagonal are not iid, but because their contribution is so small if $N\rightarrow\infty$, the circular law will still be approximately valid [[3]](https://arxiv.org/abs/1012.4818).

Finally, to find the eigenvalues of the Jacobian, the eigenvalues $\mu_i$ of $J$ are translated by 1 to find the eigenvalues $\lambda_i=\mu_i-1$ of $J-I$. Therefore, the eigenvalues of $J-I$ are located on a circle of radius $g$ centered at $-1$. The first time an eigenvalue has a positive real part then is when $g=1$, which signifies the change in stability in the fixed point at the origin and also corresponds to the bifurcation.
