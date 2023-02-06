import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd

# import params
from parameters import *

# load data
df = pd.read_csv("data.csv")

# prepare the plot
fig, ax = plt.subplots()
x = np.linspace(0, L, 100)
X, Y = np.meshgrid(x, x)
Z = potential.compute_values(X, Y)
plt.imshow(Z, extent=[0, L, 0, L], origin="lower", cmap="viridis_r", alpha=0.5)
plt.colorbar()
qv = ax.quiver(
    df.iloc[0, 0:N],
    df.iloc[0, N : 2 * N],
    np.cos(df.iloc[0, 4 * N : 5 * N]),
    np.sin(df.iloc[0, 4 * N : 5 * N]),
    color="grey",
    clim=[-np.pi, np.pi],
    headaxislength=10,
    headlength=9,
)
ax.set_xlim(0, L)
ax.set_ylim(0, L)
plt.tight_layout()


def animate(i):
    x = df.iloc[i, 0:N]
    y = df.iloc[i, N : 2 * N]
    vx = df.iloc[i, 2 * N : 3 * N].to_numpy()
    vy = df.iloc[i, 3 * N : 4 * N].to_numpy()
    vel_norm = np.sqrt((vx**2 + vy**2))
    vx, vy = vx / vel_norm, vy / vel_norm
    pos = np.vstack((x, y))
    qv.set_offsets(pos.T)
    qv.set_UVC(vx, vy)
    return qv


anim = FuncAnimation(fig, animate, np.arange(1, Tmax), interval=100)

plt.show()