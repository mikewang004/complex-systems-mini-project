import numpy as np 
import matplotlib.pyplot as plt
import scipy as sp
import Exercise1 as exc1

dt = 0.02
sigma = 10; b = 8/3; rho = 28; h = 0.01; 

Dr = 30



def dr_dt(x, sigma = sigma, b = b, rho = rho, h = h): # Lorenz system 
    x_new = np.zeros(3)
    x_new[0] = sigma * (x[1] - x[0]) #dx/dt
    x_new[1] = rho * x[0] - x[1] - x[0] * x[2] #dy/dt
    x_new[2] = x[0] * x[1] - b * x[2] #dz/dt

    return x_new 

def runge_kutta(x, sigma = sigma, b = b, rho = rho, h = h):
    k1 = h * dr_dt(x, sigma, b, rho, h)
    k2 = h * dr_dt(x + k1/2, sigma, b, rho, h)
    k3 = h * dr_dt(x + k2/2, sigma, b, rho, h)
    k4 = h * dr_dt(x + k3)
    x_new = x + k1/6 + k2/3 + k3/3 + k4/6 

    return x_new 


def simulate_system(n = 1000, h = h, x_0 = np.array([1, 0, 0])):
    x_array = np.zeros([3, n]);
    x_array[:, 0] = x_0
    for i in range(0, n-1):
        x_array[:, i+1] = runge_kutta(x_array[:, i], h = h)
    return x_array 


class NeuralNetwork():
    def __init__():
        pass