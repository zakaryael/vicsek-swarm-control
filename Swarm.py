import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import KDTree
import json
import csv
from potentials import *


class Swarm:
    def __init__(
        self,
        positions,
        orientations,
        vel_magnitude,
        sensing_radius,
        noise,
        box_length,
        potential_fields=None,
        save_json=False,
        save_csv=True,
    ):
        """initializes the swarm object
        parameters:
            positions (dxN numpy array): Holds the coordinates of the N particles forming the swarm
            orientations (array of size N): The orientations of the N particles
            vel_magnitude (float): initial velocity magnitude of the particles
            sensing_radius (float):
            noise (float):
            box_length (float):
        """
        assert orientations.shape[0] == positions.shape[1]
        self.size = orientations.shape[0]
        self.positions = positions
        self.orientations = orientations
        self.vel_magnitude = vel_magnitude
        self.velocities = vel_magnitude * np.vstack(
            (np.cos(orientations), np.sin(orientations))
        )
        self.sensing_radius = sensing_radius
        self.noise = noise
        self.box_length = box_length  ## not so happy about box_length being here
        self.neighbors = None
        self.potential_fields = potential_fields
        self.iteration = 0
        self.save_json = save_json
        self.save_csv = save_csv

    def compute_neighbors(self):
        """finds the particles with the sensing radius of every particle in the swarm"""
        self.neighbors = {}
        for particle, position in enumerate(self.positions.T):
            for particle_, position_ in enumerate(self.positions.T):
                dist_squarred = np.sum((position - position_) ** 2)
                if dist_squarred < self.sensing_radius**2:
                    if particle in self.neighbors:
                        self.neighbors[particle].append(particle_)
                    else:
                        self.neighbors[particle] = [particle_]
        return self.neighbors

    def compute_orientations(self):
        for key, value in self.neighbors.items():
            self.orientations[key] = self.orientations[
                value
            ].mean() + self.noise * np.random.uniform(-np.pi, np.pi, size=1)
        return self.orientations

    def update_orientations(self):
        # change into :update orientations to average orientations
        tree = KDTree(self.positions.T)  # , boxsize=[self.box_length, self.box_length]
        dist = tree.sparse_distance_matrix(
            tree, max_distance=self.sensing_radius, output_type="coo_matrix"
        )
        data = np.exp(self.orientations[dist.col] * 1j)
        neigh = sparse.coo_matrix((data, (dist.row, dist.col)), shape=dist.get_shape())
        S = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1)))
        self.orientations = np.angle(S) + self.noise * np.random.uniform(
            -np.pi, np.pi, size=self.size
        )

    def update_positions(self):
        vx, vy = np.cos(self.orientations), np.sin(self.orientations)
        if self.potential_fields is None:
            velocity = self.vel_magnitude * np.vstack((vx, vy))
            self.positions += velocity
        else:
            v_potentials = np.zeros((2, self.size))
            for potential in self.potential_fields.values():
                v_potentials += potential.compute_gradients(self.positions)
            velocity = (
                self.vel_magnitude * np.vstack((vx, vy))
                + v_potentials
                + self.noise * np.random.uniform(-1, 1, size=(2, self.size))
            )  
            self.positions += velocity
            vx, vy = velocity[0], velocity[1]

        self.positions[self.positions < 0] += self.box_length
        self.positions[self.positions > self.box_length] -= self.box_length

        vx, vy = vx / np.linalg.norm(velocity, axis=0), vy / np.linalg.norm(
            velocity, axis=0
        )
        self.vx = vx
        self.vy = vy
        self.velocities = velocity

    def evol(self):
        self.update_orientations()
        self.update_positions()

        if self.save_json:
            self.save_to_json()

        if self.save_csv:
            self.save_to_csv()
        self.iteration += 1

    def compute_order_param(self):
        return np.linalg.norm(self.velocities.mean(axis=1)) / self.vel_magnitude

    def update_potential(self, control_potential):
        self.potential_fields["control"] = control_potential

    def save_to_json(self):
        """
        saves the swarm information to a json file
        """
        # check if the file already exists:

        # if exists append the file with the current values.

        info = {
            "positions": self.positions.tolist(),
            "orientations": self.orientations.tolist(),
            "velocities": self.velocities.tolist(),
            "order_param": self.compute_order_param(),
        }
        ## check if file exists with
        # if path.isfile(fname) is False: raise Exception("file not found")
        if self.iteration == 0:
            with open("data.json", "w") as f:
                json.dump({self.iteration: info}, f)

        else:
            with open("data.json", "r") as f:
                data = json.load(f)

            data[self.iteration] = info

            with open("data.json", "w") as f:
                json.dump(data, f, indent=4, separators=(",", ": "))

    def save_to_csv(self):
        """saves the swarm data to a csv file"""
        # 1st step construct the headers row
        headers = (
            [f"x_{i}" for i in range(self.size)]
            + [f"y_{i}" for i in range(self.size)]
            + [f"vx_{i}" for i in range(self.size)]
            + [f"vy_{i}" for i in range(self.size)]
            + [f"theta_{i}" for i in range(self.size)]
            + ["order_param"]
        )

        # create a list of the values to save:
        data = (
            list(self.positions.flatten())
            + list(self.velocities.flatten())
            + list(self.orientations)
            + [self.compute_order_param()]
        )

        # create the file if it's the first iteration:
        if self.iteration == 0:
            with open("data.csv", "w") as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                writer.writerow(data)

        # on subsequent iterations, append to the existing csv file
        else:
            with open("data.csv", "a") as file:
                writer = csv.writer(file)
                writer.writerow(data)

    def load_from_json(self):
        pass

    def load_from_csv(self):
        pass
    
    def in_range(self, position, radius):
        dist = np.linalg.norm(self.positions - position, axis=0)
        return dist < radius
