import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import csv

from matplotlib.backends.backend_agg import FigureCanvasAgg


class SwarmVisualizer:
    def __init__(self, L, cmap="RdYlBu_r", walls=None):
        self.cmap = cmap
        self.L = L
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        #add walls
        if walls is not None:
            for wall in walls:
                if wall["axis"] == 0:
                    #add horizontal line at y = wall["origin"][1] spanning from wall["origin"][0] to wall["origin"][0] + wall["length"]
                    self.ax.axhline(y=wall["origin"][1], xmin=wall["origin"][0]/self.L, xmax=(wall["origin"][0] + wall["length"])/self.L, color="black", linewidth=3)
                else:
                    #add vertical line at x = wall["origin"][0] spanning from wall["origin"][1] to wall["origin"][1] + wall["length"]
                    self.ax.axvline(x=wall["origin"][0], ymin=wall["origin"][1]/self.L, ymax=(wall["origin"][1] + wall["length"])/self.L, color="black", linewidth=3)
        # , self.ax = plt.subplot(figsize=(10, 10))
        # self.ax.set_xlim(0, self.L)
        # self.ax.set_ylim(0, self.L)
        # self.ax.set_aspect("equal")
        # self.ax.set_xlabel("x")
        # self.ax.set_ylabel("y")
        # self.ax.set_title("Vicsek Swarm")
        # self.ax.set_facecolor("white")
        # self.ax.set_axisbelow(True)
        self.qv = None
        self.fields = None
        # # Show the graph without blocking the rest of the programrenderer
        
        plt.show(block=False)

    def _render_swarm(self, positions, orientations):
        # plot the swarm
        if self.qv is None:
            self.qv = self.ax.quiver(
                positions[0],
                positions[1],
                np.cos(orientations),
                np.sin(orientations),
                #orientations,
                scale=60,
                #scale_units="xy",
                color="white",
                width=0.005,
                headwidth=3,
                headlength=5,
                headaxislength=3,
                #clim=[-np.pi, np.pi],
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
        # # return the image as a 3D numpy array
        # img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((int(self.fig.bbox.bounds[3]), int(self.fig.bbox.bounds[2]), -1))
    
        # return img
    
    #create a function to render using data from a csv file row by row
    def render_from_csv(self, csv_file, N, potential_fields):
        """Renders the swarm from a csv file row by row.
        Args:
            csv_file (str): The path to the csv file.
            Returns:
                img (np.array): The image as a 3D numpy array."""
        import time
        with open(csv_file, 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                #check if the row contains float values
                if not all(isinstance(x, float) for x in row):
                    continue
                row = np.array(row) # consits of 2 * N positions, N orientations, 2 target positions, 2 control positions, 1 time
                positions = row[:2 * N].reshape((2, N))
                orientations = row[2 * N:3 * N]
                target = row[3 * N:3 * N + 2]
                print(f"target: {target}")
                control = row[3 * N + 2:3 * N + 4]
                time = row[3 * N + 4]
                #check that control is the same as potential_fields["control"].loc
                assert np.allclose(control, potential_fields["control"].loc)
                #render the swarm
                self.render(positions, orientations, potential_fields, target, time)

                time.sleep(1)




        
