# complex-systems-mini-project

We chose the project High-Dimensional Dynamical Systems:

An interesting toy model for understanding the brain and an example of a high-dimensional dynamical system comes from the dynamics of a randomly wired network of nonlinear firing-rate “neurons”. The dynamics of the state, $r_i$ for $i= 1,2, . . . , N$, is given by the equation
$$\tau \frac{dr_i}{dt}=−r_i+\sum_{j=1}^{N}J_{ij}φ(r_j) \text{ (13)}$$

wherer $i$ is the activity of unit $i$ and $\tau$ is a relaxation time constant which we assume is the same across units. The coupling matrix $J_{ij}$ encodes interactions among different units and we model those interactions as a random matrix, each $J_{ij}$ is an independent draw from a Gaussian distribution $J_{ij}∼N(0, g^2/N)$, that is a Gaussian distribution with zero mean and standard deviation $σ=g/\sqrt{N}$. Remove any self-coupling by setting $J_{ii}=0$ and take the transfer function as $φ(x)=tanh(x)$.

1. Using a relatively large population of units (e.g. $N∼100$), use numerical simulations to show that the network dynamics undergo a bifurcation at $g∼1$. Describe the dynamics below and above this bifurcation.
2. Linearize the system around the no-activity fixed point $\vec{r}=0$. Use results from random matrix theory to analytically identify the bifurcation point as $g=1$ in the thermodynamic limit.
3. In a recent paper [[8]](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.5.043044), the tools from dynamical systems were used to exactly compute the Lyapunov spectrum and the attractor dimension (through the Kaplan-Yorke conjecture) of this high-dimension system. Reproduce their results through Fig. 4.
4. Recurrent Networks such as that given in Eq. 13 have been recently used as powerful tools for time-series prediction such as in Pathak et al [[9]](https://arxiv.org/abs/1710.07313).Implement this prediction algorithm for a chaotic Lorenz system. How good are your predictions? What could you change to make them better? See me for further references once you have made substantial progress.