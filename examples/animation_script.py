import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from parameters import *
from Swarm import *
from potentials import *
from matplotlib.animation import FuncAnimation

#a function to plot the potential fields
def plot_potential_fields(potential_fields, box_length, ax):
    x = np.linspace(0, box_length, 100)
    X, Y = np.meshgrid(x, x)
    # save the potential plots into a tuple
    potential_plots = []
    for potential in potential_fields.values():
        Z = potential.compute_values(X, Y)
        potential_plots.append(ax.imshow(Z, extent=[0, box_length, 0, box_length], origin='lower',
                    cmap='RdYlBu_r', alpha=0.5))
        #add a legend()
        #ax.legend(potential_plots, potential_fields.keys())

    return tuple(potential_plots)



# set the initial positions and orientations of the particles
np.random.seed(0)
pos = np.random.uniform(0,box_length,size=(2,N))
orient = np.random.uniform(-np.pi, np.pi,size=N)


# set the figure
fig = plt.figure(figsize=(10,10), dpi=100, facecolor='w', edgecolor='k')

ax0 = fig.add_subplot(1, 2, 1)
qv = ax0.quiver(pos[0], pos[1], np.cos(orient), np.sin(orient), color = 'grey', clim=[-np.pi, np.pi], headaxislength=10, headlength=9)
pots = plot_potential_fields(potential_fields, box_length, ax0)

def init():
    ax0.set_xlim(0, box_length)
    ax0.set_ylim(0, box_length)
    return qv, pots[0], pots[1]

swarm = Swarm(pos, orient, v0, r0, eta, box_length, potential_fields)

def animate(i):
    potential_fields["control"].update_location(potential_fields["control"].loc + np.array([0.01, 0.01]))
    swarm.update_potential(potential_fields["control"])
    swarm.evol()
    qv.set_offsets(swarm.positions.T)
    qv.set_UVC(swarm.vx, swarm.vy)
    #pots[0].set_array(potential_fields["control"].compute_values(X, Y))
    pots = plot_potential_fields(potential_fields, box_length, ax0)
    return qv, pots[0], pots[1]

anim = FuncAnimation(fig, animate, init_func=init, frames=10000, interval=1, blit=True)
# anim.save('../animation.gif', writer='imagemagick', fps=60)

plt.show()