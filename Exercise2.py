import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# N = 100
# mu = 0
# tau = 1.0
h = 0.01
# steps = 1000

rng = np.random.default_rng()


def generate_J(N, mu, sigma):
    J = np.random.normal(mu, sigma, (N, N))
    np.fill_diagonal(J, 0)
    return J


def phi(x):
    return np.tanh(x)


# alternative to dr_dt: iterative instead of every r_i in one go
# def dr_dt_iterative(r, J, tau):
#     drdt = np.zeros(N)
#     for i in range(N):
#         sum_j = np.sum(J[i, :] * phi(r))
#         drdt[i] = (-r[i] + sum_j) / tau
#     return drdt


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


def run_simulation_orthonormal_starts(N, steps, tau, mu, g):
    sigma = g / np.sqrt(N)
    J = generate_J(N, mu, sigma)

    x_all = np.zeros((steps, N, N))

    x = np.eye(N)  # Each column of x is an orthonormal starting position

    for start_pos in range(N):
        x_start = x[:, start_pos]

        for i in range(steps):
            x_start = runge_kutta(x_start, J, tau)  # Update the state using Runge-Kutta
            x_all[i, :, start_pos] = x_start  # Store the result for this position

    return x_all


def qr_decomposition_timesteps(x_all, steps):
    R_matrices = []

    for i in range(steps):
        matrix_at_timestep = x_all[i, :, :]

        Q, R = sp.linalg.qr(matrix_at_timestep)

        R_matrices.append(R)

    return R_matrices


def exponents_spectrum(R_matrices, steps):
    m = R_matrices[0].shape[0]

    diagonal_sum = np.zeros(m)

    for i in range(steps):
        R = R_matrices[i]
        diagonal_sum += np.log(np.abs(np.diag(R)))

    exponents = diagonal_sum / steps

    return exponents


def variable_g_max_exponent(num_gs):
    max_exponents = []
    for g in np.linspace(0.5, 100, num_gs):
        x_all = run_simulation_orthonormal_starts(50, 1000, 1, 0, g)
        R_matrices = qr_decomposition_timesteps(x_all, 1000)
        # for i in R_matrices:
        #     print(i.shape)
        diagonal_avg = exponents_spectrum(R_matrices, 1000)
        max_exponents.append(max(diagonal_avg))
    return np.linspace(0.5, 100, num_gs), max_exponents


def variable_g_sim(g_arr, N, steps, tau, mu):
    x_norm_arr = np.zeros((len(g_arr), steps))
    for j, g in enumerate(g_arr):
        x_norm_arr[j, :] = run_simulation(N, steps, tau, mu, g)
    return x_norm_arr


if __name__ == "__main__":

    # g_arr = np.array([0.9, 1.0, 1.1])

    # x_norm_arr = variable_g_sim(g_arr, N, steps, tau, mu)

    # for j in range(len(g_arr)):
    #     plt.plot(x_norm_arr[j], label=f"g = {g_arr[j]}")
    # plt.legend()
    # plt.xlabel("Time steps")
    # plt.ylabel("Norm of activity vector")
    # plt.title("Neural Network Dynamics for Different g")
    # plt.show()

    x_all = run_simulation_orthonormal_starts(100, 1000, 1, 0, 1000)
    R_matrices = qr_decomposition_timesteps(x_all, 1000)
    # for i in R_matrices:
    #     print(i.shape)
    diagonal_avg = exponents_spectrum(R_matrices, 1000)

    plt.plot(diagonal_avg)

    # x_axis, max_exponents = variable_g_max_exponent(20)

    # plt.plot(x_axis, max_exponents)

    plt.show()
