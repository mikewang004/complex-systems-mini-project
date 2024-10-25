import numpy as np 
import matplotlib.pyplot as plt
import scipy as sp

dt = 0.02



def lorenz(x, y, z):
    dxdt = 10*(y - x)
    dydt = x*(28 -z) -y
    dzdt = x*y -8 *z/3


