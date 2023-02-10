#create an environment

from env import SwarmEnv
from potentials import GaussianPotential, InverseSquarePotential
from Swarm import Swarm
from pg_learner import *

loc_target = 7. * np.ones(2)
loc_control = 3. * np.ones(2)
gaussian_potential_params = {"alpha": 1, "A": 0.2}
inverse_potential_params = {"b": 1, "n": 1, "A": 0.02}
target_potential = GaussianPotential(loc_target, gaussian_potential_params)
control_potential = InverseSquarePotential(loc_control, inverse_potential_params)
potential_fields = {"target": target_potential, "control": control_potential}

env = SwarmEnv(N=10, L=10, v0=1, r0=1, eta=0.1, potential_fields=potential_fields, Tmax=10000, target_radius=1)

#initialize policy and value networks
policy_net = PolicyNetwork(env.observation_spec().shape[0], env.action_spec().num_values, [4])
value_net = ValueNetwork(env.observation_spec().shape[0], [4])

#initialize learner
learner = PolicyGradientLearner(env, policy_net, value_net, lr_policy=0.01, lr_value=0.01)

#train
learner.train(20)