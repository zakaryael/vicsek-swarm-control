import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg


class SwarmVisualizer:
    def __init__(self, L, cmap="RdYlBu_r"):
        self.cmap = cmap
        self.L = L
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(0, self.L)
        self.ax.set_ylim(0, self.L)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title("Vicsek Swarm")
        self.ax.set_facecolor("white")
        self.ax.set_axisbelow(True)
        self.qv = None
        self.fields = None
        # Show the graph without blocking the rest of the program
        plt.show(block=False)

    def _render_swarm(self, positions, orientations):
        # plot the swarm
        if self.qv is None:
            self.qv = self.ax.quiver(
                positions[0],
                positions[1],
                np.cos(orientations),
                np.sin(orientations),
                orientations,
                scale=60,
                scale_units="xy",
                color="blue",
                width=0.005,
                headwidth=3,
                headlength=5,
                headaxislength=3,
                clim=[-np.pi, np.pi],
            )
        else:
            self.qv.set_offsets(positions.T)
            self.qv.set_UVC(np.cos(orientations), np.sin(orientations))

    def _render_potential_fields(self, potential_fields):
        # plot the potential fields
        if self.fields is None:
            self.x = np.linspace(0, self.L, 100)
            self.y = np.linspace(0, self.L, 100)
            self.X, self.Y = np.meshgrid(self.x, self.y)

        Z = np.zeros_like(self.X)
        for potential in potential_fields.values():
            Z += potential.compute_values(self.X, self.Y)

        if self.fields is None:
            self.fields = self.ax.imshow(
                Z,
                cmap=self.cmap,
                alpha=1,
                extent=[0, self.L, 0, self.L],
                origin="lower",
                interpolation="none",
                aspect="auto",
                vmin=Z.min(),
                vmax=Z.max(),
            )

        else:
            self.fields.set_data(Z)

    def _render_target(self, target):
        # plot the target
        self.ax.scatter(
            target[0], target[1], s=100, color="red", marker="X", label="target"
        )

    def render(self, positions, orientations, potential_fields, target, iteration):
        # render the swarm, the potential fields and the target
        self._render_swarm(positions, orientations)
        self._render_potential_fields(potential_fields)
        self._render_target(target)
        # add the iteration number to the title
        self.ax.set_title(f"Vicsek Swarm - Time {iteration} s")
        
        # update the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # return the image as a 3D numpy array
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((int(self.fig.bbox.bounds[3]), int(self.fig.bbox.bounds[2]), -1))
        

        return img
        

