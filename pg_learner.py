import torch
import torch.nn as nn
from torch.nn.functional import relu, softmax
import numpy as np


class PolicyNetwork:
    def __init__(self, obs_dim, act_dim, hidden_layers):
        self.network_layers = [obs_dim] + hidden_layers + [act_dim]
        self.n_layers = len(hidden_layers) + 2
        self.parameters = []
        self.g = torch.Generator().manual_seed(42)
        self.policy = torch.empty(obs_dim, act_dim)

        for layer in range(self.n_layers - 1):
            self.parameters.append(
                torch.randn(
                    self.network_layers[layer],
                    self.network_layers[layer + 1],
                    requires_grad=True,
                )
            )

    def update_policy(self, obs):
        """Update the policy for a given observation."""
        out = relu(obs @ self.parameters[0])
        for i in range(1, self.n_layers - 2):
            out = relu(out @ self.parameters[i])
        self.policy[obs] = softmax(out @ self.parameters[-1], dim=1)
    
    def get_action(self, obs):
        """Sample an action from the policy for a given observation."""
        return torch.multinomial(self.policy[obs], num_samples=1, generator=self.g).item()

    def update_params(self, parameters):
        self.parameters = parameters

    def get_action_prob(self, obs, act):
        return self.policy[obs, act]


class ValueNetwork:
    def __init__(self, obs_dim, hidden_layers=[4]):
        self.network_layers = [obs_dim] + hidden_layers + [1]
        self.n_layers = len(hidden_layers) + 2
        self.parameters = []
        for layer in range(self.n_layers - 1):
            self.parameters.append(
                torch.randn(
                    self.network_layers[layer],
                    self.network_layers[layer + 1],
                    requires_grad=True,
                )
            )

    def compute_value(self, obs):
        out = relu(obs @ self.parameters[0])
        for i in range(1, self.n_layers - 2):
            out = relu(out @ self.parameters[i])
        return out @ self.parameters[-1]

    def update_params(self, parameters):
        self.parameters = parameters


class PolicyGradientLearner:
    def __init__(self, env, policy_net, value_net, lr_policy, lr_value):
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.lr_policy = lr_policy
        self.lr_value = lr_value
    
    def train(self, n_episodes):
        for episode in range(n_episodes):
            time_step = self.env.reset()
            obs = torch.tensor(time_step.observation)
            done = False
            for iter in range(self.env.Tmax):
                ##___ forward pass policy___##
                self.policy_net.update_policy(obs)
                act = self.policy_net.get_action(obs) #this is where the action is sampled from the policy
                prob = self.policy_net.get_action_prob(obs, act) #this is where the probability of the action is computed
                ##____________________##
                
                time_step = self.env.step(act) # this is where  we take the action and get the next time step
                new_obs = torch.tensor(time_step.observation) #this is where the observation is updated
                reward = time_step.reward #this is where the reward is updated
                done = time_step.last() # check if the episode is done
                ##___ forward pass value network ___##
                v = self.value_net.compute_value(obs)
                with torch.no_grad():
                    if done:
                        v_new = 0
                    else:
                        v_new = self.value_net.compute_value(new_obs) #this might not be the most efficient way things, but let's worry about that later
                ##____________________##
                delta = reward + v_new - v #this is the TD error

                ##___update value network parameters___##
                v.backward()
                for p in self.value_net.parameters:
                    p += self.lr_value * delta * p.grad
                    p.grad.zero_()
                ##____________________##

                ##___update policy network parameters___##
                log_prob = torch.log(prob)
                log_prob.backward()
                for p in self.policy_net.parameters:
                    p += self.lr_policy * delta * p.grad
                    p.grad.zero_()
                ##____________________##
                obs = new_obs
                if done:
                    break



#create an environment
loc_target = 3 * np.ones(2)
loc_control = 7 * np.ones(2)
gaussian_potential_params = {"alpha": 1, "A": 0.02}
inverse_potential_params = {"b": 1, "n": 1, "A": 0.02}
target_potential = GaussianPotential(loc_target, gaussian_potential_params)
control_potential = InverseSquarePotential(loc_control, inverse_potential_params)
potential_fields = {"target": target_potential, "control": control_potential}

env = SwarmEnv(N=10, L=10, v0=1, r0=1, eta=0.1, potential_fields=potential_fields, Tmax=100, target_radius=1)

#initialize policy and value networks
policy_net = PolicyNetwork(env.observation_spec().shape[0], env.action_spec().num_values, [4])
value_net = ValueNetwork(env.observation_spec().shape[0], [4])

#initialize learner
learner = PolicyGradientLearner(env, policy_net, value_net, lr_policy=0.01, lr_value=0.01)

#train
learner.train(10)


            


