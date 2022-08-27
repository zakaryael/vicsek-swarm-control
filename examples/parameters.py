
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Swarm import Potential, Swarm
import numpy as np

#parameters
L = 10.0
rho = 10.0
N = int(rho*L**2)

r0 = 0.5
deltat = 1.0
factor =0.5
v0 = 0.03#r0/deltat*factor
eta = 0.2

loc = 3 * np.ones(2)
potential = Potential(loc, 0.03, 1)

Tmax = 100000