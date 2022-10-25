
import sys
sys.path.insert(0, '/Users/zakaryaelkhiyati/code/vicsek-swarm-control/')
from Swarm import *
from parameters import *
from tqdm import tqdm

np.random.seed(0)
swarm = Swarm(np.random.uniform(0,L,size=(2,N)), np.random.uniform(-np.pi, np.pi,size=N), v0, r0, eta, L, potential=potential)

print(f"N is {N}")
for i in tqdm(range(Tmax)):
    swarm.evol()


    
