# This file will have methods to compute the potential field gradients and values for different potentials

import numpy as np


class Potential:
    def __init__(self, loc):
        """initializes the potential"""
        self.loc = loc
    
    def compute_distance_vector(self, positions):
        """computes the distance vector between the potential and the agents
        parameters:
            positions (dxN numpy array): the positions of the agents
        """
        self.dist_vec = positions - self.loc[:, np.newaxis]

    def update_location(self, increment, boxsize, boundary_conditions):
        """updates the location of the potential"""
        self.loc += increment
        if boundary_conditions == "periodic":
            self.loc[self.loc < 0] += boxsize
            self.loc[self.loc > boxsize] -= boxsize
        
        elif boundary_conditions == "reflective":
            if self.loc[0] < 0 or self.loc[0] > boxsize:
                increment[0] *= -1
            if self.loc[1] < 0 or self.loc[1] > boxsize:
                increment[1] *= -1
            self.loc += increment
        


# define a class for gaussian potential that inherits from the potential class
class GaussianPotential(Potential):
    """gaussian potential
    for a potential of the form A*exp(-alpha*r^2)
    parameters:
        loc (dx1 numpy array): location of the potential
        params (dictionary): dictionary of parameters for the potential
            A (float): amplitude of the potential
            alpha (float): scale parameter
    """

    def __init__(self, loc, params):
        super().__init__(loc)
        self.A = params["A"]
        self.alpha = params["alpha"]

    def compute_gradients(self, positions):
        """Computes the potential gradients at given locations
        parameters:
            positions (dxN numpy array): locations at which to compute the potential
        """
        self.compute_distance_vector(positions)
        return (
            -2
            * self.alpha
            * self.dist_vec
            * self.A
            * np.exp(-self.alpha * np.sum(self.dist_vec ** 2, axis=0))
        )

    def compute_values(self, x, y):
        """computes the values of the potential at position (x,y)"""
        return self.A * np.exp(
            -self.alpha * ((x - self.loc[0]) ** 2 + (y - self.loc[1]) ** 2)
        )


class InverseSquarePotential(Potential):
    """inverse square potential
    for a potential of the form A/(b^2 + r^2)^n
    parameters:
        loc (dx1 numpy array): location of the potential
        params (dictionary): dictionary of parameters for the potential
            A (float): amplitude of the potential
            b (float): scale parameter
            n (float): power of the potential
    """

    def __init__(self, loc, params):
        super().__init__(loc)
        self.A = params["A"]
        self.b = params["b"]
        self.n = params["n"]

    def compute_values(self, x, y):
        return (
            self.A
            / (self.b**2 + (x - self.loc[0]) ** 2 + (y - self.loc[1]) ** 2) ** self.n
        )

    def compute_gradients(self, positions):
        self.compute_distance_vector(positions)
        return (
            -2
            * self.A
            * self.n
            * self.dist_vec
            / (self.b**2 + np.sum(self.dist_vec ** 2, axis=0))
            ** (self.n + 1)
        )

class LennardJonesPotential(Potential):
    """Lennard-Jones potential
    for a potential of the form 4*epsilon*((sigma/r)**12 - (sigma/r)**6)
    parameters:
        loc (dx1 numpy array): location of the potential
        params (dictionary): dictionary of parameters for the potential
            epsilon (float): depth of the potential well
            sigma (float): distance at which the inter-particle potential is zero
    """

    def __init__(self, loc, params):
        super().__init__(loc)
        self.epsilon = params["epsilon"]
        self.sigma = params["sigma"]

    def compute_gradients(self, positions):
        # compute distance vector between particles
        self.compute_distance_vector(positions)
        dist_sq = np.sum(self.dist_vec ** 2, axis=0)

        # compute the force magnitudes using the Lennard-Jones potential
        force_mag = 24 * self.epsilon * (
            2 * (self.sigma ** 12 / dist_sq ** 7) - (self.sigma ** 6 / dist_sq ** 4)
        )

        # compute the force directions
        force_dir = self.dist_vec / np.sqrt(dist_sq)

        # compute the force components
        force_components = force_mag * force_dir

        # sum over all particles to get the net force on each particle
        net_force = np.sum(force_components, axis=0)

        return net_force

    def compute_values(self, x, y):
        r = np.sqrt((x - self.loc[0]) ** 2 + (y - self.loc[1]) ** 2)
        return 4 * self.epsilon * ((self.sigma / r) ** 12 - (self.sigma / r) ** 6)

class CosinePotential(Potential):
    """cosine potential
    for a potential of the form A*cos(w*r + phi)
    parameters:
        loc (dx1 numpy array): location of the potential
        params (dictionary): dictionary of parameters for the potential
            A (float): amplitude of the potential
            w (float): frequency of the potential
            phi (float): phase of the potential
    """

    def __init__(self, loc, params):
        super().__init__(loc)
        self.A = params["A"]
        self.w = params["w"]
        self.phi = params["phi"]

    def compute_values(self, x, y):
        r = np.sqrt((x - self.loc[0]) ** 2 + (y - self.loc[1]) ** 2)
        return self.A * np.cos(self.w * r + self.phi)

    def compute_gradients(self, positions):
        self.compute_distance_vector(positions)
        r = np.sqrt(np.sum(self.dist_vec ** 2, axis=0))
        return (
            -self.A
            * self.w
            * np.sin(self.w * r + self.phi)
            * self.dist_vec
            / r
        )

class InverseDistancePotential(Potential):
    """Inverse distance potential
    for a potential of the form A/r
    parameters:
        loc (dx1 numpy array): location of the potential
        params (dictionary): dictionary of parameters for the potential
            A (float): amplitude of the potential
    """
    def __init__(self, loc, params):
        super().__init__(loc)
        self.A = params["A"]

    def compute_values(self, x, y):
        r = np.sqrt((x - self.loc[0]) ** 2 + (y - self.loc[1]) ** 2)
        return self.A / r if r > 0 else np.inf

    def compute_gradients(self, positions):
        self.compute_distance_vector(positions)
        r = np.sqrt(np.sum(self.dist_vec ** 2, axis=0))
        return -self.A * self.dist_vec / r**3 if r > 0 else np.zeros_like(self.dist_vec)


class YukawaPotential(Potential):
    """Yukawa potential
    for a potential of the form A * exp(-k*r) / r
    parameters:
        loc (dx1 numpy array): location of the potential
        params (dictionary): dictionary of parameters for the potential
            A (float): amplitude of the potential
            k (float): decay constant of the potential
    """
    def __init__(self, loc, params):
        super().__init__(loc)
        self.A = params["A"]
        self.k = params["k"]

    def compute_values(self, x, y):
        r = np.sqrt((x - self.loc[0]) ** 2 + (y - self.loc[1]) ** 2)
        return self.A * np.exp(-self.k * r) / r

    def compute_gradients(self, positions):
        self.compute_distance_vector(positions)
        r = np.sqrt(np.sum(self.dist_vec ** 2, axis=0))
        return -self.A * (self.k*r + 1) * self.dist_vec * np.exp(-self.k*r) / r**3
    
