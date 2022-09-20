import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from parameters import *
from Swarm import *

np.random.seed(0)
pos = np.random.uniform(0,L,size=(2,N))
orient = np.random.uniform(-np.pi, np.pi,size=N)


from matplotlib.animation import FuncAnimation

fig, ax= plt.subplots()
x = np.linspace(0, L, 100)
X, Y = (np.meshgrid(x,x))
Z = potential.compute2(X, Y)        
plt.imshow(Z, extent=[0, L, 0, L], origin='lower',
                cmap='viridis_r', alpha=0.5)
plt.colorbar()
qv = ax.quiver(pos[0], pos[1], np.cos(orient), np.sin(orient), color = 'grey', clim=[-np.pi, np.pi], headaxislength=10, headlength=9)

def init():
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    plt.tight_layout()
    return qv
swarm = Swarm(pos, orient, v0, r0, eta, L, potential)

def animate(i):
    swarm.evol()
    qv.set_offsets(swarm.positions.T)
    qv.set_UVC(swarm.vx, swarm.vy)
    return qv

anim = FuncAnimation(fig,animate,init_func=init, frames=100, interval=100)
anim.save('../animation.gif', writer='imagemagick', fps=60)

plt.show()