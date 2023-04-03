
from parameters import *
from src.Swarm import *
from tqdm import tqdm

np.random.seed(0)
swarm = Swarm(np.random.uniform(0,L,size=(2,N)), np.random.uniform(-np.pi, np.pi,size=N), v0, r0, eta, L, potential_fields, boundary_conditions="periodic")

print(f"{N=}")

for i in tqdm(range(Tmax)):
    swarm.evol()