import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


N = 100
g = 1
mu = 0
tau = 1.0
h = 0.01
rng = np.random.default_rng()


def generate_J(N, mu, sigma):
    J = np.random.normal(mu, sigma, (N, N))
    np.fill_diagonal(J, 0)
    return J


def phi(x):
    return np.tanh(x)




def dr_dt(r, phi, J, tau):
    drdt = np.zeros(N)
    for i in range(N):
        sum_j = np.sum(J[i, :] * phi(r))  
        drdt[i] = (-r[i] + sum_j) / tau
        r[i] = drdt[i]
    return drdt


def runge_kutta(x, phi, J, tau):
    k1 = h * dr_dt(x, phi, J, tau)
    k2 = h * dr_dt(x + k1 / 2, phi, J, tau)
    k3 = h * dr_dt(x + k2 / 2, phi, J, tau)
    k4 = h * dr_dt(x + k3, phi, J, tau)
    x_new = x + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
    return x_new


steps = 1000

def run_simulation(N,steps, phi, tau, mu, g):
    sigma = g / np.sqrt(N)
    x = np.random.uniform(-1, 1, N).T
    J = generate_J(N, mu, sigma**2)
    x_norm = np.zeros([steps])

    for i in range(0, steps):
        x = runge_kutta(x, phi, J, tau)
        x_norm[i] = np.linalg.norm(x)

    return x_norm

def variable_g_sim(g_arr, x_norm_arr, N, steps, phi, tau, mu, g):
    for j in range(0, size_g_arr):
        x_norm_arr[j, :] = run_simulation(N, steps, phi, tau, mu, g_arr[j])

    return x_norm_arr

size_g_arr = 3
g_arr = np.array([0.1, 1, 10])
x_norm_arr = np.zeros([size_g_arr, steps])



#x_norm = run_simulation(N, steps, phi, tau, mu, g)
x_norm_arr = variable_g_sim(g_arr, x_norm_arr, N, steps, phi, tau, mu, g)
for j in range(0, size_g_arr):
    plt.plot(x_norm_arr[j], label = "g = " + str(g_arr[j]))
    plt.legend()
plt.show()
