# This file will have methods to compute the potential field gradients and values for different potentials
# Path: potentials.py

import numpy as np


class Potential:
    def __init__(self, loc, params):
        """initializes the potential"""
        pass

    def compute_gradients(self, positions):
        """Computes the potential gradients at given locations
        parameters:
            positions (dxN numpy array): locations at which to compute the potential
        """
        pass

    def compute_values(self, x, y):
        """computes the values of the potential at position (x,y)"""
        pass

    def update_location(self, loc):
        self.loc = loc


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
        self.loc = loc
        self.A = params["A"]
        self.alpha = params["alpha"]

    def compute_gradients(self, positions):
        return (
            -2
            * self.alpha
            * (positions.T - self.loc).T
            * self.A
            * np.exp(-self.alpha * np.sum((positions.T - self.loc) ** 2, axis=1))
        )

    def compute_values(self, x, y):
        return self.A * np.exp(
            -self.alpha * ((x - self.loc[0]) ** 2 + (y - self.loc[1]) ** 2)
        )


# define a class for the generalized inverse square potential that inherits from the potential class
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
        self.loc = loc
        self.A = params["A"]
        self.b = params["b"]
        self.n = params["n"]

    def compute_values(self, x, y):
        # positions = np.array([x,y])
        return (
            self.A
            / (self.b**2 + (x - self.loc[0]) ** 2 + (y - self.loc[1]) ** 2) ** self.n
        )
        # return self.A/(self.b**2 + np.sum((positions.T - self.loc) ** 2, axis=1))**self.n

    def compute_gradients(self, positions):
        return (
            -2
            * self.A
            * self.n
            * (positions.T - self.loc).T
            / (self.b**2 + np.sum((positions.T - self.loc) ** 2, axis=1))
            ** (self.n + 1)
        )
