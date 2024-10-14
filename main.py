import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


N = 100
g = 2
mu = 0
sigma = g / np.sqrt(N)
tau = 1
h = 0.01
rng = np.random.default_rng()


def generate_J(N, mu, sigma):
    J = np.random.normal(mu, sigma, (N, N))
    np.fill_diagonal(J, 0)
    return J


def phi(x):
    return np.tanh(x)


def dr_dt(r, phi, J, tau):
    return (-r + J @ phi(r)) / tau


def runge_kutta(x, phi, J, tau):
    k1 = h * dr_dt(x, phi, J, tau)
    k2 = h * dr_dt(x + k1 / 2, phi, J, tau)
    k3 = h * dr_dt(x + k2 / 2, phi, J, tau)
    k4 = h * dr_dt(x + k3, phi, J, tau)
    x_new = x + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
    return x_new


steps = 1000

x = np.random.randint(0, 2, N).T
J = generate_J(N, mu, sigma)
x_norm = np.zeros([steps])

for i in range(0, steps):
    x = runge_kutta(x, phi, J, tau)
    x_norm[i] = np.linalg.norm(x)


plt.plot(x_norm)
plt.show()
