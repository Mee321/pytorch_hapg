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

class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(Actor, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
        self.dist = DiagGaussian(hidden_size, num_outputs)
        self.train()

    def forward(self, x):
        hidden_actor = self.actor(x)
        return hidden_actor

    def act(self, inputs):
        actor_features = self.actor(inputs)
        dist = self.dist(actor_features)
        action = dist.sample()
        action_log_probs = dist.log_probs(action)
        return action, action_log_probs, None

    def evaluate_action(self, inputs, action):
        actor_features = self.actor(inputs)
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action)
        return action_log_probs, None

class Critic(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(Critic, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.train()

    def forward(self, x):
        hidden_critic = self.critic(x)
        return self.critic_linear(hidden_critic)

    def get_value(self, inputs):
        return self.forward(inputs)