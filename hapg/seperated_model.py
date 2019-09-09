import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hapg.distributions import Bernoulli, Categorical, DiagGaussian
from hapg.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(Policy, self).__init__()

        self.dist = DiagGaussian(hidden_size, num_outputs)
        self.affine1 = nn.Linear(num_inputs, hidden_size)
        self.affine2 = nn.Linear(hidden_size, hidden_size)

    def act(self, inputs, deterministic=False):
        actor_features = self.forward(inputs)
        dist = self.dist(actor_features)
        action = dist.sample()
        dist_entropy = dist.entropy().mean()

        action_log_probs = dist.log_probs(action)
        return action, action_log_probs, dist_entropy

    def evaluate_action(self, inputs, action):
        actor_feature = self.forward(inputs)
        dist = self.dist(actor_feature)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return action_log_probs, dist_entropy

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))
        return x


class Value(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, hidden_size)
        self.affine2 = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        state_values = self.value_head(x)
        return state_values

    def get_value(self, inputs):
        return self.forward(inputs)