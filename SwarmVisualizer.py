import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import csv
import matplotlib.colors as colors
from matplotlib.backends.backend_agg import FigureCanvasAgg


class SwarmVisualizer:
    def __init__(self, L, cmap="RdYlBu_r", walls=None, quiver=False):
        self.L = L
        self.fig = plt.figure(figsize=[10, 10])
        self.ax = self.fig.add_subplot(111)
        self.quiver = quiver
        self.scp = None
        # add walls
        if walls is not None:
            for wall in walls:
                if wall["axis"] == 0:
                    self.ax.axhline(
                        y=wall["origin"][1],
                        xmin=wall["origin"][0],
                        xmax=(wall["origin"][0] + wall["length"]),
                        color="gray",
                        linewidth=2,
                    )
                else:
                    self.ax.axvline(
                        x=wall["origin"][0],
                        ymin=wall["origin"][1],
                        ymax=(wall["origin"][1] + wall["length"]),
                        color="gray",
                        linewidth=2,
                    )

        self.ax.set_xlim(0, self.L)
        self.ax.set_ylim(0, self.L)

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        self.ax.set_facecolor("white")
        self.ax.set_axisbelow(True)
        self.qv = None
        self.fields = None
        # Create the colormap using a ListedColormap object
        cmap = colors.ListedColormap(
            ["#C5E0DC", "#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
        )
        self.cmap = cmap
        # Create a normalization object that maps data values to colormap range
        self.norm = colors.Normalize(vmin=0, vmax=0.01)
        self.fig.colorbar(
            mappable=mpl.cm.ScalarMappable(norm=self.norm, cmap=self.cmap),
            ax=self.ax,
            label="Potential Field",
        )
        plt.show(block=False)

    def _render_swarm(self, positions, orientations):
        # plot the swarm
        if self.quiver is True:
            if self.qv is None:
                self.qv = self.ax.quiver(
                    positions[0],
                    positions[1],
                    np.cos(orientations),
                    np.sin(orientations),
                    orientations,
                    scale=60,
                    scale_units="xy",
                    color="black",
                    width=0.005,
                    headwidth=3,
                    headlength=5,
                    headaxislength=3,
                    clim=[-np.pi, np.pi],
                )
            else:
                self.qv.set_offsets(positions.T)
                self.qv.set_UVC(np.cos(orientations), np.sin(orientations))
        else:
            # plot the swarm as a scatter plot
            if self.scp is None:
                self.scp = self.ax.scatter(
                    positions[0],
                    positions[1],
                    s=20,
                    color="black",
                    marker="o",
                    label="swarm",
                )
            else:
                self.scp.set_offsets(positions.T)

    def _render_potential_fields(self, potential_fields):
        # plot the potential fields
        if self.fields is None:
            self.x = np.linspace(0, self.L, 1000)
            self.y = np.linspace(0, self.L, 1000)
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
                norm=self.norm,
            )

        else:
            self.fields.set_data(Z)

    def _render_target(self, target, target_radius):
        # plot the target
        self.ax.scatter(
            target[0], target[1], s=100, color="red", marker="X", label="target"
        )
        # draw a circle around the target with radius r
        circle = plt.Circle(
            (target[0], target[1]),
            radius=target_radius,
            color="red",
            fill=False,
            linestyle="dashed",
        )
        self.ax.add_artist(circle)

    def render(self, positions, orientations, potential_fields, target, target_radius):
        # render the swarm, the potential fields and the target
        self._render_swarm(positions, orientations)
        self._render_potential_fields(potential_fields)
        self._render_target(target, target_radius)
        # update the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # # return the image as a 3D numpy array
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
            (int(self.fig.bbox.bounds[3]), int(self.fig.bbox.bounds[2]), -1)
        )
        return img

    def set_title(self, title):
        self.ax.set_title(title)

    # create a function to render using data from a csv file row by row
    def render_from_csv(self, csv_file, N, potential_fields, target_radius):
        """Renders the swarm from a csv file row by row.
        Args:
            csv_file (str): The path to the csv file.
            Returns:
                img (np.array): The image as a 3D numpy array."""
        import time

        with open(csv_file, "r") as f:
            csv_reader = csv.reader(f)
            numeric_data = False
            for row in csv_reader:
                if not numeric_data:
                    numeric_data = True

                else:
                    row = np.array(
                        row
                    )  # consits of 2 * N positions, N orientations, 2 target positions, 2 control positions, 1 time step
                    positions = row[: 2 * N].reshape((2, N))
                    orientations = row[2 * N : 3 * N]
                    target = row[3 * N : 3 * N + 2]
                    control = row[3 * N + 2 : 3 * N + 4]
                    iteration = row[3 * N + 4]
                    self.render(
                        positions,
                        orientations,
                        potential_fields,
                        target,
                        target_radius,
                    )
    
    def close(self):
        plt.close(self.fig)
