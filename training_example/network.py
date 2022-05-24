import numpy as np
import torch
from torch.distributions import Normal


class Actor(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()

        # defining fully-connected layers
        self.fc_1 = torch.nn.Linear(input_size, hidden_size)
        self.fc_2 = torch.nn.Linear(hidden_size, hidden_size)
        self.mu_layer = torch.nn.Linear(hidden_size, output_size)
        self.std_layer = torch.nn.Linear(hidden_size, output_size)

    # get action for given observation
    def forward(self, observation, deterministic=False, with_logprob=True):
        net_out = torch.relu(self.fc_1(observation))
        net_out = self.fc_2(net_out)
        mu = self.mu_layer(net_out)
        log_std = self.std_layer(net_out)

        # minimum log standard deviation is choosen as -20
        # maximum log standard deviation is choosen as +2
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        # pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - torch.nn.functional.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        # only %60 of the steering command will be used
        steer = 0.6 * pi_action[:, 0].reshape(-1, 1)

        # acceleration is from 0 ti 1;  braking is from 0 to -1
        accel_brake = pi_action[:, 1].reshape(-1, 1)

        # apply tangent hyperbolic activation functions to actions
        steer = torch.tanh(steer)
        accel_brake = torch.tanh(accel_brake)

        pi_action= torch.cat((steer, accel_brake), 1)
        return pi_action, logp_pi


class Critic(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()

        # defining fully-connected layers
        self.fc_1 = torch.nn.Linear(input_size - 2, hidden_size)
        self.fc_2 = torch.nn.Linear(hidden_size + output_size, hidden_size)
        self.fc_3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_4 = torch.nn.Linear(hidden_size, 1)

    # get value for given state-action pair
    def forward(self, state, action):
        out = self.fc_1(state)
        out = torch.nn.functional.relu(out)
        out = torch.nn.functional.relu(self.fc_2(torch.cat([out, action], 1)))
        out = torch.nn.functional.relu(self.fc_3(out))
        out = self.fc_4(out)
        return out
