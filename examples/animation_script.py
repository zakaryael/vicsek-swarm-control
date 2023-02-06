import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from parameters import *
from Swarm import *
from potentials import *
from matplotlib.animation import FuncAnimation



# set the initial positions and orientations of the particles
np.random.seed(0)
pos = np.random.uniform(0,L,size=(2,N))
orient = np.random.uniform(-np.pi, np.pi,size=N)


# set the figure
fig = plt.figure(figsize=(10,10), dpi=100, facecolor='w', edgecolor='k')

ax0 = fig.add_subplot(1, 2, 1)

x = np.linspace(0, L, 100)
X, Y = (np.meshgrid(x,x))
Z = potential.compute_values(X, Y) 
qv = ax0.quiver(pos[0], pos[1], np.cos(orient), np.sin(orient), color = 'grey', clim=[-np.pi, np.pi], headaxislength=10, headlength=9)
pot = ax0.imshow(Z, extent=[0, L, 0, L], origin='lower',
            cmap='viridis_r', alpha=0.5)
def init():

    ax0.set_xlim(0, L)
    ax0.set_ylim(0, L)
    return qv, pot

swarm = Swarm(pos, orient, v0, r0, eta, L, potential)

def animate(i):
    potential.update_location(potential.loc + np.array([0.01, 0.01]))
    swarm.update_potential(potential)
    swarm.evol()
    qv.set_offsets(swarm.positions.T)
    qv.set_UVC(swarm.vx, swarm.vy)
    Z = potential.compute_values(X, Y) 
    pot = ax0.imshow(Z, extent=[0, L, 0, L], origin='lower',
                cmap='viridis_r', alpha=0.5)
    return qv, pot

anim = FuncAnimation(fig, animate, init_func=init, frames=10000, interval=1, blit=True)
# anim.save('../animation.gif', writer='imagemagick', fps=60)

plt.show()