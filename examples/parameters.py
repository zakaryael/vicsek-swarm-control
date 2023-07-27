import os, sys
project_dir = (os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add the parent directory to the path
sys.path.append(project_dir) # add the parent directory to the path

import pickle

from src.Potentials import *
import numpy as np
#parameters
L = 1.0
rho = 1000.0
N = int(rho*L**2)

r0 = 0.05
v0 = 1e-2
eta = 0.1
repulsion = 0

boundary_conditions = "reflective"

WALLS = [{"origin": np.array([0.4, 0.0]), "length": 0.45,  "axis": 1}, {"origin": np.array([0.4, 0.55]), "length": 0.45,  "axis": 1}, {"origin": np.array([0.0, 0.4]), "length": 0.45,  "axis": 0}, {"origin": np.array([0.55, 0.4]), "length": 0.45,  "axis": 0}]
maze_path = os.path.join(project_dir, "data", "maze6bis.pckl")
with open(maze_path, 'rb') as f:
    maze = pickle.load(f)

WALLS = maze

loc_target = 0.7 * np.ones(2)
loc_control = 0.3 * np.ones(2) #np.random.rand(2)

# default values gaussian_potential_params = {"alpha": 100, "A": 0.01}

gaussian_potential_params = {"alpha": 3000, "A": 0.01}
gaussian_potential_params2 = {"alpha": 200, "A": 0.01}

lj_potential = LennardJonesPotential(loc=np.array([0.7, 0.3]), params={"sigma": 0.001, "epsilon": 10})


loc = np.array([0.5, 0.5])
params = {"A": 0.01, "w": 20, "phi": 0}
cos_potential = CosinePotential(loc, params)
#instantiating a YukawaPotential object
yukawa_potential = YukawaPotential(loc_target, {"A": 0.02, "k": 100})

inverse_potential_params = {"b": 2, "n": 3, "A": 0.01}
inverse_potential = InverseSquarePotential(loc=np.array([0.7, 0.3]), params=inverse_potential_params)
target_potential = GaussianPotential(loc_target, gaussian_potential_params)
control_potential = GaussianPotential(loc_control, gaussian_potential_params2)
potential_fields = {"control": control_potential}

Tmax = 400

target_radius=0.04

env_params = dict(N=N, L=L, v0=v0, r0=r0, eta=eta, potential_fields=potential_fields, Tmax=Tmax, target_radius=target_radius, boundary_conditions=boundary_conditions, walls=WALLS, repulsion=repulsion, target_location=loc_target, coefficient_of_restitution=1, seed=np.random.randint(0, 1000000))