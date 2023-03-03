
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add the parent directory to the path
import pickle

from potentials import *
import numpy as np

#parameters
L = 1.0
rho = 1000.0
N = int(rho*L**2)

r0 = 0.05
v0 = 0.01
eta = 0.1

boundary_conditions = "reflective"

WALLS = [{"origin": np.array([0.4, 0.0]), "length": 0.45,  "axis": 1}, {"origin": np.array([0.4, 0.55]), "length": 0.45,  "axis": 1}, {"origin": np.array([0.0, 0.4]), "length": 0.45,  "axis": 0}, {"origin": np.array([0.55, 0.4]), "length": 0.45,  "axis": 0}]

with open('../data/maze2.pckl', 'rb') as f:
    maze = pickle.load(f)

WALLS = maze

loc_target = 0.7 * np.ones(2)
loc_control = 0.1 * np.ones(2)

# default values gaussian_potential_params = {"alpha": 100, "A": 0.01}

gaussian_potential_params = {"alpha": 1000, "A": 0.01}
gaussian_potential_params2 = {"alpha": 100, "A": 0.01}

inverse_potential_params = {"b": 0.1, "n": 2, "A": 0.0000}
target_potential = GaussianPotential(loc_target, gaussian_potential_params)
control_potential = GaussianPotential(loc_control, gaussian_potential_params)
potential_fields = {"target": target_potential, "control": control_potential}

Tmax = 1000

target_radius=0.01