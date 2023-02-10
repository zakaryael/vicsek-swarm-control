
from parameters import *
from Swarm import *
from potentials import *
from tqdm import tqdm

np.random.seed(0)
swarm = Swarm(np.random.uniform(0,box_length,size=(2,N)), np.random.uniform(-np.pi, np.pi,size=N), v0, r0, eta, box_length, potential=potential)

print(f"{N=}")

for i in tqdm(range(Tmax)):
    swarm.evol()