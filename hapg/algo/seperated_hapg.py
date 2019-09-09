import torch
import torch.nn as nn
import torch.optim as optim
from hapg.utils import *

# LVC version, DiCE with a bug when denominator becomes 0
class HAPG_LVC():
    def __init__(self,
                 actor,
                 critic,
                 value_loss_coef,
                 entropy_coef,
                 critic_lr=None,
                 actor_lr=None,
                 max_grad_norm=None):

        self.actor = actor
        self.critic = critic
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.actor_lr = actor_lr
        self.max_grad_norm = max_grad_norm
        # self.alignment = sum([len(p.view(-1)) for p in list(self.actor_critic.parameters())[4:10]])
        # value net optimizer
        self.optimizer = optim.Adam(
            critic.parameters(), lr=critic_lr)

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values = self.critic(rollouts.obs[:-1].view(-1, *obs_shape))
        action_log_probs, dist_entropy = self.actor.evaluate_action(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)
        probs = torch.exp(action_log_probs)

        magic_box = probs/probs.detach()
        action_loss = -(advantages.detach() * magic_box).mean()
        grad = torch.autograd.grad(action_loss, self.actor.parameters(), retain_graph=True)
        grad = flatten(grad)

        prev_params = get_flat_params_from(self.actor)
        direction = grad / torch.norm(grad)
        updated_params = prev_params - self.actor_lr * direction
        d_theta = updated_params - prev_params
        set_flat_params_to(self.actor, updated_params)

        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

        return value_loss.item(), action_loss.item(), None, grad, d_theta

    def inner_update(self, rollouts, prev_grad, d_theta):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values = self.critic(rollouts.obs[:-1].view(-1, *obs_shape))
        action_log_probs, dist_entropy = self.actor.evaluate_action(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)
        probs = torch.exp(action_log_probs)

        magic_box = probs / probs.detach()
        action_loss = -(magic_box * advantages.detach()).mean()
        jacob = torch.autograd.grad(action_loss, self.actor.parameters(), retain_graph=True, create_graph=True)
        jacob = flatten(jacob)
        product = torch.dot(jacob, d_theta)
        d_grad = torch.autograd.grad(product, self.actor.parameters(), retain_graph=True)
        grad = prev_grad + flatten(d_grad)

        # update params
        prev_params = get_flat_params_from(self.actor)
        direction = grad / torch.norm(grad)
        updated_params = prev_params - self.actor_lr * direction
        d_theta = updated_params - prev_params
        set_flat_params_to(self.actor, updated_params)

        # action_loss = -(advantages * action_log_probs).mean()
        # print(torch.autograd.grad(action_loss, self.actor_critic.parameters(), allow_unused=True))

        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

        return value_loss.item(), action_loss.item(), None, grad, d_theta
