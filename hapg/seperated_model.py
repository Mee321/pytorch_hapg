import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from hapg.distributions import Bernoulli, Categorical, DiagGaussian
from hapg.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, hidden_size)
        self.affine2 = nn.Linear(hidden_size, hidden_size)
        self.action_mean = nn.Linear(64, num_outputs)
        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        torch.nn.init.orthogonal_(self.affine1.weight)
        torch.nn.init.orthogonal_(self.affine2.weight)
        torch.nn.init.orthogonal_(self.action_mean.weight)

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))
        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        return action_mean, action_log_std, None

    def act(self, inputs):
        action_mean, action_log_std, _ = self.forward(inputs)
        with torch.no_grad():
            action = torch.normal(action_mean, torch.exp(action_log_std))
        var = torch.exp(action_log_std) ** 2
        action_log_probs = -((action - action_mean) ** 2) / (2 * var) - action_log_std - math.log(math.sqrt(2 * math.pi))
        action_log_probs = action_log_probs.sum(1, keepdim=True)
        # None for placeholder
        return action, action_log_probs, None

    def evaluate_action(self, inputs, action):
        action_mean, log_std, _ = self.forward(inputs)
        var = torch.exp(log_std) ** 2
        action_log_probs = -((action - action_mean) ** 2) / (2 * var) - log_std - math.log(math.sqrt(2 * math.pi))
        action_log_probs = action_log_probs.sum(1, keepdim=True)
        # None for placeholder
        return action_log_probs, None


class Value(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, hidden_size)
        self.affine2 = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)
        torch.nn.init.orthogonal_(self.affine1.weight)
        torch.nn.init.orthogonal_(self.affine2.weight)
        torch.nn.init.orthogonal_(self.value_head.weight)

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        state_values = self.value_head(x)
        return state_values

    def get_value(self, inputs):
        return self.forward(inputs)