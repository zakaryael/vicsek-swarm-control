import torch
import torch.nn as nn
from torch.nn.functional import relu, softmax
import numpy as np
from tqdm import tqdm

class PolicyNetwork:
    def __init__(self, obs_dim, act_dim, hidden_layers):
        self.network_layers = [obs_dim] + hidden_layers + [act_dim]
        self.n_layers = len(hidden_layers) + 2
        self.parameters = []
        self.g = torch.Generator().manual_seed(42)
        self.probs = torch.empty(1, act_dim)

        for layer in range(self.n_layers - 1):
            self.parameters.append(
                torch.randn(
                    self.network_layers[layer],
                    self.network_layers[layer + 1],
                    requires_grad=True,
                    dtype=torch.double,
                )
            )

    def update_policy(self, obs):
        """Update the policy for a given observation."""
        out = relu(obs @ self.parameters[0])
        for i in range(1, self.n_layers - 2):
            out = relu(out @ self.parameters[i])
        self.probs = softmax(out @ self.parameters[-1], dim=1)
    
    def get_action(self, obs):
        """Sample an action from the policy for a the current observation."""
        return torch.multinomial(self.probs, num_samples=1, generator=self.g).item()

    def update_params(self, parameters):
        self.parameters = parameters

    def get_action_prob(self, act):
        return self.probs[0, act]


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
                    dtype=torch.double,
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
        print(f"____ start of training over {n_episodes=} ____")
        for episode in range(n_episodes):
            print(f"____ start of {episode=} ____")
            time_step = self.env.reset()
            obs = torch.tensor(time_step.observation).reshape(1, -1)
            #for iter in tqdm(range(self.env.Tmax)):
            while True:
                ##___ forward pass policy___##
                self.policy_net.update_policy(obs)
                act = self.policy_net.get_action(obs) #this is where the action is sampled from the policy
                prob = self.policy_net.get_action_prob(act) #this is where the probability of the action is computed
                ##____________________##
                
                time_step = self.env.step(act) # this is where  we take the action and get the next time step
                new_obs = torch.tensor(time_step.observation).reshape(1, -1) #this is where the observation is updated
                reward = time_step.reward #this is where the reward is updated
                ##___ forward pass value network ___##
                v = self.value_net.compute_value(obs)
                with torch.no_grad():
                    if self.env.done():
                        v_new = 0
                    else:
                        v_new = self.value_net.compute_value(new_obs) #this might not be the most efficient way things, but let's worry about that later
                ##____________________##
                delta = reward + v_new - v #this is the TD error

                ##___update value network parameters___##
                v.backward()
                for p in self.value_net.parameters:
                    p.data += self.lr_value * delta * p.grad
                    p.grad.zero_()
                ##____________________##

                ##___update policy network parameters___##
                log_prob = torch.log(prob)
                log_prob.backward()
                for p in self.policy_net.parameters:
                    p.data += self.lr_policy * delta * p.grad
                    p.grad.zero_()
                ##____________________##
                obs = new_obs
                if self.env.done():
                    print(f"____ end of {episode=} ____")
                    print(f"{reward=}")
                    break
                


            


