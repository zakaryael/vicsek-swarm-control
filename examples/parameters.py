
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add the parent directory to the path

from potentials import *
import numpy as np

#parameters
L = 10.0
rho = 2
N = int(rho*L**2)

r0 = 0.5
v0 = 0.01
eta = 0.002

loc = 3 * np.ones(2)
gaussian_potential_params = {"alpha": 1, "A": 0.02}
inverse_potential_params = {"b": 1, "n": 3, "A": 0.02}
potential = GaussianPotential(loc, gaussian_potential_params)
potential = InverseSquarePotential(loc, inverse_potential_params)

Tmax = 100000