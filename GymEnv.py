from Swarm import Swarm

from SwarmVisualizer import *
import copy
import csv

import gym
import numpy as np
from gym import spaces
import os

class SwarmEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(
        self,
        potential_fields,
        N=1000,
        L=10,
        v0=0.1,
        r0=0.1,
        eta=0.05,
        Tmax=1000,
        repulsion=0,
        boundary_conditions="periodic",
        target_location=None,
        target_radius=0.1,
        walls=None,
        control_amplitude=0.01,
        out_dir=None,
        seed=42,
        coefficient_of_restitution=1,
    ):
        super(SwarmEnv, self).__init__()
        self.g = np.random.default_rng(seed=seed)  # random number generator
        self.boundary_conditions = boundary_conditions
        self.walls = walls
        self.repulsion = repulsion
        self.out_dir = out_dir
        self.target_location = target_location if target_location is not None else potential_fields["target"].loc
        self.target_radius = target_radius
        self.N = N
        self.L = L
        self.v0 = v0
        self.r0 = r0
        self.eta = eta
        self.potential_fields = copy.deepcopy(potential_fields)
        self.Tmax = Tmax
        self.iteration = 0  # iteration counter for the episode
        self.visualization = None
        self.n_trapped = 0
        self.n_trapped_old = 0
        self.coefficient_of_restitution = coefficient_of_restitution
        # Define action and observation space
        # start with the action space: continuous 2d vector
        self.action_space = spaces.Box(
            low=-control_amplitude, high=control_amplitude, shape=(2,), dtype=np.float64
        )  #
        # now the observation space: continuous (3 * N + 5) vector
        lower_bounds = np.concatenate(
            (
                np.zeros(2 * N),
                -np.pi * np.ones(N),
                np.zeros(2),
                np.array([0, 0]),
                np.zeros(1),
            )
        )
        upper_bounds = np.concatenate(
            (
                self.L * np.ones(2 * N),
                np.pi * np.ones(N),
                self.L * np.ones(2),
                np.array([self.L, self.L]),
                self.Tmax * np.ones(1),
            )
        )
        self.observation_space = spaces.Box(
            low=lower_bounds, high=upper_bounds, shape=(3 * N + 5,), dtype=np.float64
        )  # low=lower_bounds, high=upper_bounds,
        # now the swarm
        self.swarm = Swarm(
            self.g.uniform(0, self.L, size=(2, self.N)),
            self.g.uniform(-np.pi, np.pi, size=N),
            v0,
            r0,
            eta,
            L,
            potential_fields=potential_fields,
            boundary_conditions=boundary_conditions,
            walls=walls,
            repulsion=repulsion,
            coefficient_of_restitution=coefficient_of_restitution,
        )

    def step(self, action):
        # compute the number of agents in the target
        self.n_trapped = np.sum(
            self.swarm.in_range(
                self.target_location, self.target_radius
            ),
            dtype=np.float32,
        )
        self.swarm.update_control_potential(action)
        self.swarm.evol()
 
        # compute the reward
        reward = self.reward()
        reward = float(reward)
        # check if the episode is done
        done = self.done()
        done = bool(done)

        # compute the observation
        observation = self.observation()
        self.iteration += 1

        return observation, reward, done, {"info": "informativo"}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.iteration = 0
        self.swarm = Swarm(
            self.g.uniform(0, self.L, size=(2, self.N)),
            self.g.uniform(-np.pi, np.pi, size=self.N),
            self.v0,
            self.r0,
            self.eta,
            self.L,
            potential_fields=self.potential_fields,
            boundary_conditions=self.boundary_conditions,
            walls=self.walls,
            repulsion=self.repulsion,
            coefficient_of_restitution=self.coefficient_of_restitution,
        )
        return self.observation()

    def render(self, mode="plot", close=True):
        
        if self.out_dir:
            # Render the current state of the environment to a csv file
            fname = os.path.join(self.out_dir, "swarm.csv")
            # create the file if it doesn't exist
            if not os.path.exists(fname):
                #make a list of the header, consits of 2 self.N positions, self.N orientations, 2 target positions, 2 control positions, 1 time
                header = [f"pos_{i}" for i in range(2 * self.N)] + [f"ori_{i}" for i in range(self.N)] + [f"target_{i}" for i in range(2)] + [f"control_{i}" for i in range(2)] + ["time"]
                with open(fname, "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
            # write the current state to the file
            with open(fname, "a") as f:
                writer = csv.writer(f)
                writer.writerow(self.observation())

        if mode == "plot":
            # Render the environment to the screen
            if self.visualization is None:
                self.visualization = SwarmVisualizer(self.L, walls=self.walls)
            title = f"Vicsek Swarm - Time {self.iteration} s \n Captured: {self.n_trapped} \n Captured on last step: {self.n_captured()}"
            self.visualization.set_title(title)

            img = self.visualization.render(
                self.swarm.positions,
                self.swarm.orientations,
                self.potential_fields,
                self.target_location,
                self.target_radius,
            )
            if self.done():
                self.close()
            return img
        



    def close(self):
        """closes the environment and the visualization
        """
        if self.visualization is not None:
            self.visualization.close()

    def observation(self):
        """returns the observation
        returns:
            observation (numpy array): the observation
        """
        return np.concatenate(
            (
                self.swarm.positions.flatten(),
                self.swarm.orientations.flatten(),
                self.target_location,
                self.potential_fields["control"].loc,
                np.array([self.Tmax - self.iteration]),
            ),
            dtype=np.float32,
        )

    def reward(self):
        """returns the reward
        returns:
            reward (float): the reward
        """
        reward = - np.linalg.norm(self.swarm.positions - self.target_location[:, np.newaxis], axis=0).mean()
        if self.done():
            reward += self.n_trapped
        return reward
    
    def n_captured(self):
        """returns the number of agents newly in the target"""
        n_captured = np.max(self.n_trapped - self.n_trapped_old, 0)
        self.n_trapped_old = self.n_trapped
        return n_captured

    def done(self):
        """returns if the episode is done
        returns:
            done (bool): if the episode is done
        """
        return self.iteration >= self.Tmax - 1 or self.n_trapped >= self.swarm.size