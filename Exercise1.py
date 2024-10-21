import numpy as np
import matplotlib.pyplot as plt

N = 100
mu = 0
tau = 1.0
h = 0.01
steps = 1000

rng = np.random.default_rng()


def generate_J(N, mu, sigma):
    J = np.random.normal(mu, sigma, (N, N))
    np.fill_diagonal(J, 0)
    return J


def phi(x):
    return np.tanh(x)


# alternative to dr_dt: iterative instead of every r_i in one go
def dr_dt_iterative(r, J, tau):
    drdt = np.zeros(N)
    for i in range(N):
        sum_j = np.sum(J[i, :] * phi(r))
        drdt[i] = (-r[i] + sum_j) / tau
    return drdt


def dr_dt(r, J, tau):
    drdt = (-r + np.dot(J, phi(r))) / tau
    return drdt


def runge_kutta(x, J, tau):
    k1 = h * dr_dt(x, J, tau)
    k2 = h * dr_dt(x + k1 / 2, J, tau)
    k3 = h * dr_dt(x + k2 / 2, J, tau)
    k4 = h * dr_dt(x + k3, J, tau)
    x_new = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x_new


def run_simulation(N, steps, tau, mu, g):
    sigma = g / np.sqrt(N)
    x = np.random.uniform(-0.01, 0.01, N)
    J = generate_J(N, mu, sigma)
    x_norm = np.zeros(steps)

    for i in range(steps):
        x = runge_kutta(x, J, tau)
        x_norm[i] = np.linalg.norm(x)

    return x_norm


def variable_g_sim(g_arr, N, steps, tau, mu):
    x_norm_arr = np.zeros((len(g_arr), steps))
    for j, g in enumerate(g_arr):
        x_norm_arr[j, :] = run_simulation(N, steps, tau, mu, g)
    return x_norm_arr


g_arr = np.array([0.9, 1.0, 1.1])

x_norm_arr = variable_g_sim(g_arr, N, steps, tau, mu)

for j in range(len(g_arr)):
    plt.plot(x_norm_arr[j], label=f"g = {g_arr[j]}")
plt.legend()
plt.xlabel("Time steps")
plt.ylabel("Norm of activity vector")
plt.title("Neural Network Dynamics for Different g")
plt.show()
