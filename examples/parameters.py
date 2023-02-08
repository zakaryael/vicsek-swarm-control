
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add the parent directory to the path

from potentials import *
import numpy as np

#parameters
box_length = 10.0
rho = 10
N = int(rho*box_length**2)

r0 = 0.5
v0 = 0.01
eta = 0.002

loc_target = 3 * np.ones(2)
loc_control = 7 * np.ones(2)
gaussian_potential_params = {"alpha": 1, "A": 0.02}
inverse_potential_params = {"b": 1, "n": 1, "A": 0.02}
target_potential = GaussianPotential(loc_target, gaussian_potential_params)
control_potential = InverseSquarePotential(loc_control, inverse_potential_params)
potential_fields = {"target": target_potential, "control": control_potential}

Tmax = 100000