# in this file we will use dm_env to create the environment for the swarm

import dm_env
import numpy as np
from Swarm import Swarm
from potentials import *
import matplotlib.pyplot as plt

"""the learning task will be to make the swarm move towards a target  starting from a random initial configuration but in a specific region of the 2d box
the target will be a gaussian potential with a fixed location and a fixed strength
the swarm will be controlled by a second potential field with a moving location and a fixed strength
the reward of each step will be the number of particles in the swarm that are within a certain distance of the target
the episode will end when all the particles are within a certain distance of the target or when the maximum number of iterations is reached
the observation will be the positions of the particles in the swarm, the position of the target, the position of the control potential, and the remaining number of iterations
the action will be the position of the control potential"""

# define the environment class
class SwarmEnv(dm_env.Environment):
    """Swarm Environment
    This environment is a 2D environment with a swarm of N agents
    The agents are initialized with random positions and orientations"""
    # define the action space
    Actions = [np.ones(2), np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1]), np.zeros(2), -np.ones(2)] # discrete action space with 7 actions temporarily for testing

    def __init__(self, N, L, v0, r0, eta, potential_fields, Tmax, target_radius):
        """initializes the swarm environment
        parameters:
            N (int): number of agents in the swarm
            L (float): length of the environment
            v0 (float): initial velocity of the agents
            r0 (float): sensing radius of the agents
            eta (float): noise of the agents
            potential (Potential): potential field to be used
            Tmax (int): maximum number of iterations per episode
        """
        self.N = N
        self.L = L
        self.v0 = v0
        self.r0 = r0
        self.eta = eta
        self.potential_fields = potential_fields
        self.Tmax = Tmax
        self.iteration = 0 # iteration counter for the episode
        self.reset_next_step = False # flag to reset the environment at the next step
        self.swarm = Swarm(
            np.random.uniform(0, L, size=(2, N)),
            np.random.uniform(-np.pi, np.pi, size=N),
            v0,
            r0,
            eta,
            L,
            potential_fields=potential_fields,
        )
        self.target_radius = target_radius

    def reset(self):
        """resets the environment and returns the initial observation
        returns:
            observation (dm_env.TimeStep): the initial observation
        """
        self.iteration = 0
        self.swarm = Swarm(
            np.random.uniform(0, self.L, size=(2, self.N)),
            np.random.uniform(-np.pi, np.pi, size=self.N),
            self.v0,
            self.r0,
            self.eta,
            self.L,
            potential_fields=self.potential_fields,
        )
        return dm_env.restart(self.observation())

    def step(self, action):
        """takes an action and returns the next observation, reward, done, and info
        parameters:
            action (numpy array): the action to be taken
        returns:
            observation (dm_env.TimeStep): the next observation
            reward (float): the reward for the step
        """
        self.iteration += 1
        self.potential_fields["control"].update_location(self.potential_fields["control"].loc + self.Actions[action])
        self.swarm.update_potential(self.potential_fields["control"])
        self.swarm.evol()

        observation = self.observation()
        reward = self.reward()
        done = self.done()
        info = self.info()
        return dm_env.transition(observation, reward)

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
            )
        )
    
    def reward(self):
        """returns the reward
        returns:
            reward (float): the reward
        """
        return np.sum(self.swarm.in_range(self.potential_fields["target"].loc, self.target_radius))
    
    def done(self):
        """returns whether the episode has ended
        returns:
            done (bool): whether the episode has ended
        """
        return self.iteration >= self.Tmax or self.reward() == self.N
    
    def info(self):
        """returns additional information
        returns:
            info (dict): additional information
        """
        return {}
    
    def observation_spec(self):
        """returns the observation spec
        returns:
            observation_spec (dm_env.specs.Array): the observation spec
        """
        return dm_env.specs.Array(
            shape=(2 * self.N + 4,),
            dtype=np.float32,
            name="observation",
        )

    def action_spec(self):
        """returns the action spec
        returns:
            action_spec (dm_env.specs.DiscreteArray): the action spec
        """
        return dm_env.specs.DiscreteArray(
            num_values=len(self.Actions),
            dtype=np.int32,
            name="action",
        )
    
    def close(self):
        """closes the environment"""
        pass

    def render(self):
        """renders the environment"""
        pass


    def plot_state(self):
        """plots the state of the environment"""
        # create the figure
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        # plot the potential fields using imshow and computing the potential values on a grid
        x = np.linspace(0, self.L, 100)
        y = np.linspace(0, self.L, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for potential_field in self.potential_fields.values():
            Z += potential_field.compute_values(X,Y)
        ax.imshow(Z, extent=[0, self.L, 0, self.L], origin="lower")
        qv = ax.quiver(self.swarm.positions[0], self.swarm.positions[1], self.swarm.vx, self.swarm.vy, color = 'grey', clim=[-np.pi, np.pi], headaxislength=10, headlength=9)
        plt.show()

    @staticmethod
    def create_env(N, L, v0, r0, eta, potential_fields, Tmax, target_radius):
        """creates the environment and its spec
        returns:
            env (dm_env.Environment): the environment
            env_spec (dm_env.specs.EnvironmentSpec): the environment spec
        """
        env = SwarmEnv(N, L, v0, r0, eta, potential_fields, Tmax, target_radius)
        env_spec = dm_env.specs.make_environment_spec(env)
        return env, env_spec
 

