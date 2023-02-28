import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import KDTree
import json
import csv
from potentials import *
import copy
from scipy.spatial import distance_matrix



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
        save_mode=None,
        boundary_conditions="periodic",
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
        self.g = np.random.default_rng(seed=42)
        self.sensing_radius = sensing_radius
        self.noise = noise
        self.box_length = box_length  ## not so happy about box_length being here
        self.velocities = 0
        self.neighbors = None
        self.potential_fields = copy.deepcopy(potential_fields)
        self.iteration = 0
        self.save_mode = save_mode
        self.boundary_conditions = boundary_conditions

    def _compute_interactions(self):
        # change into :update orientations to mean orientations
        tree = KDTree(self.positions.T)  # , boxsize=[self.box_length, self.box_length]
        dist = tree.sparse_distance_matrix(
            tree, max_distance=self.sensing_radius, output_type="coo_matrix"
        )
        data = np.exp(self.orientations[dist.col] * 1j)
        neigh = sparse.coo_matrix((data, (dist.row, dist.col)), shape=dist.get_shape())
        S = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1)))
        self.orientations = np.angle(S)

    def _add_noise(self):
        """adds noise to the orientations of the particles"""
        self.orientations += self.noise * self.g.uniform(
            -np.pi, np.pi, size=self.size
        )

    def _add_noisy_interactions(self):
        """adds the interaction term to the velocities of the particles"""
        vx, vy = np.cos(self.orientations), np.sin(self.orientations)
        self.velocities += self.vel_magnitude * np.vstack((vx, vy))

    def _add_pot_grads(self):
        """adds the potential gradient to the velocities of the particles"""
        pot_grads = np.zeros((2, self.size))
        if self.potential_fields is not None:
            for potential in self.potential_fields.values():
                pot_grads += potential.compute_gradients(self.positions)
        self.velocities += pot_grads

    def _compute_normalized_direction(self):
        vx, vy = np.cos(self.orientations), np.sin(self.orientations)
        return np.vstack((vx, vy))

    def _update_positions(self):
        """updates the positions of the particles"""
        self.positions += self.velocities

    def _update_orientations(self):
        normalized_velocities = self.velocities / np.linalg.norm(
            self.velocities, axis=0
        )
        self.orientations = np.arctan2(
            normalized_velocities[1], normalized_velocities[0]
        )
    
    def add_repulsive_forces(self, k):
        """Computes the repulsive forces between particles
        parameters:
            positions (dxN numpy array): positions of the particles
            k (float): constant in the repulsive potential
        """
        dist_matrix = distance_matrix(self.positions.T, self.positions.T)
        # replace the diagonal entries with 1
        dist_matrix[np.diag_indices_from(dist_matrix)] = 1
        # compute the inverse of the distance matrix
        inv_dist_matrix = 1 / dist_matrix
        dist_matrix[np.diag_indices_from(inv_dist_matrix)] = 0
        # compute the sum of the inverse distances
        sum_inv_dist_matrix = np.sum(inv_dist_matrix, axis=1)
        # compute the repulsive forces
        repulsive_forces = 2 * k * sum_inv_dist_matrix * self.positions
        self.velocities += repulsive_forces

    def _apply_boundary_conditions(self):
        if self.boundary_conditions == "periodic":
            self.positions[self.positions < 0] += self.box_length
            self.positions[self.positions > self.box_length] -= self.box_length
        
        elif self.boundary_conditions == "reflective":
            #1.update the velocities at the boundaries:
            self.velocities[0, self.positions[0] < 0] *= -1
            self.velocities[0, self.positions[0] > self.box_length] *= -1
            self.velocities[1, self.positions[1] < 0] *= -1
            self.velocities[1, self.positions[1] > self.box_length] *= -1

            #2. update the positions according to the updated velocities
            self.positions[self.positions < 0] += self.velocities[self.positions < 0]
            self.positions[self.positions > self.box_length] += self.velocities[self.positions > self.box_length]

            #3. update the orientations
            self._update_orientations()
    
    def _zero_velocities(self):
        self.velocities = np.zeros((2, self.size))

    def evol(self):
        self._zero_velocities()
        self._compute_interactions()
        self._add_noise()
        self._add_noisy_interactions()
        self._add_pot_grads()
        self.add_repulsive_forces(0.00001)
        self._update_positions()
        self._update_orientations()
        self._apply_boundary_conditions()

        if self.save_mode == "json":
            self.save_to_json()

        if self.save_mode == "csv":
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
        dist = np.linalg.norm(self.positions.T - position, axis=1)
        return dist < radius
