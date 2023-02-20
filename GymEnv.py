from Swarm import Swarm
from potentials import *
from SwarmVisualizer import *
import copy

import gym
import numpy as np
from gym import spaces


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
        target_radius=0.1,
    ):
        super(SwarmEnv, self).__init__()
        self.g = np.random.default_rng(seed=42)  # random number generator
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
        # Define action and observation space
        # start with the action space: continuous 2d vector
        self.action_space = spaces.Box(
            low=-0.01, high=0.01, shape=(2,), dtype=np.float32
        )  #
        self.n_trapped_old = 0
        # now the observation space: continuous (3 * N + 5) vector
        lower_bounds = np.concatenate(
            (
                np.zeros(2 * N),
                -np.pi * np.ones(N),
                np.zeros(2),
                np.array([-np.inf, -np.inf]),
                np.zeros(1),
            )
        )
        upper_bounds = np.concatenate(
            (
                self.L * np.ones(2 * N),
                np.pi * np.ones(N),
                self.L * np.ones(2),
                np.array([np.inf, np.inf]),
                self.Tmax * np.ones(1),
            )
        )
        self.observation_space = spaces.Box(
            low=lower_bounds, high=upper_bounds, shape=(3 * N + 5,), dtype=np.float32
        )  # low=lower_bounds, high=upper_bounds,
        # now the swarm
        self.swarm = Swarm(
            self.g.uniform(0, self.L / 2, size=(2, self.N)),
            self.g.uniform(-np.pi, np.pi, size=N),
            v0,
            r0,
            eta,
            L,
            potential_fields=potential_fields,
        )

    def step(self, action):
        # update the control potential field and evolve the swarm
        self.potential_fields["control"].update_location(
            self.potential_fields["control"].loc + action
        )
        #impose periodic boundary conditions on the control potential field
        self.potential_fields["control"].loc = np.mod(
            self.potential_fields["control"].loc, self.L
        )
        self.swarm.update_potential(self.potential_fields["control"])
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
            self.g.uniform(0, self.L / 2, size=(2, self.N)),
            self.g.uniform(-np.pi, np.pi, size=self.N),
            self.v0,
            self.r0,
            self.eta,
            self.L,
            potential_fields=self.potential_fields,
        )
        return self.observation()

    def render(self, mode="human", close=False):
        # Render the environment to the screen
        if self.visualization is None:
            self.visualization = SwarmVisualizer(self.L)
        return self.visualization.render(
            self.swarm.positions,
            self.swarm.orientations,
            self.potential_fields,
            self.potential_fields["target"].loc,
            self.iteration,
        )

    def close(self):
        pass

    def observation(self):
        """returns the observation
        returns:
            observation (numpy array): the observation
        """
        return np.concatenate(
            (
                self.swarm.positions.flatten(),
                self.swarm.orientations.flatten(),
                self.potential_fields["target"].loc.flatten(),
                self.potential_fields["control"].loc.flatten(),
                np.array([self.Tmax - self.iteration]),
            ),
            dtype=np.float32,
        )

    def reward(self):
        """returns the reward
        returns:
            reward (float): the reward
        """
        # compute the number of agents in the target
        self.n_trapped = np.sum(
            self.swarm.in_range(
                self.potential_fields["target"].loc, self.target_radius
            ),
            dtype=np.float32,
        )
        # return -np.linalg.norm(self.swarm.positions - self.potential_fields["target"].loc, axis=0).mean()
        reward = -0.25
        # # add the number of agents in the target during this step
        # reward += self.n_trapped - self.n_trapped_old

        # if we're on the last iteration, add the number of agents in the target
        if self.iteration == self.Tmax - 1:
            reward += self.n_trapped

        self.n_trapped_old = self.n_trapped
        # x_control, y_control = self.potential_fields["control"].loc
        # # check if the control potential field is inside the box
        # if x_control < 0 or x_control > self.L or y_control < 0 or y_control > self.L:
        #     reward -= 0  # penalize for being outside the box

        return reward

    def done(self):
        """returns if the episode is done
        returns:
            done (bool): if the episode is done
        """
        return self.iteration >= self.Tmax or self.n_trapped >= self.swarm.size
