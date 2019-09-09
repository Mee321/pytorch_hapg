import torch
import torch.nn as nn
import torch.optim as optim
from hapg.utils import *


# LVC version, DiCE with a bug when denominator becomes 0
class STORM_LVC():
    def __init__(self,
                 actor,
                 critic,
                 value_loss_coef,
                 entropy_coef,
                 critic_learning_rate=None,
                 actor_learning_rate_initial=None,
                 alpha_initial=1,
                 max_grad_norm=None):

        self.actor = actor
        self.critic = critic
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate_initial = actor_learning_rate_initial
        self.alpha_initial = alpha_initial
        self.max_grad_norm = max_grad_norm
        #self.alignment = sum([len(p.view(-1)) for p in list(self.actor_critic.parameters())[4:10]])
        # value net optimizer
        self.optimizer = optim.Adam(
            critic.parameters(), lr=self.critic_learning_rate)
        self.grad_norm_sq_cum = 0
        self.iteration = 1

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

        magic_box = probs / probs.detach()
        action_loss = -(advantages.detach() * magic_box).mean()
        grad = torch.autograd.grad(action_loss, self.actor.parameters(), retain_graph=True)
        grad = flatten(grad)

        prev_params = get_flat_params_from(self.actor)
        direction = grad / torch.norm(grad)
        updated_params = prev_params - self.actor_learning_rate_initial * direction
        d_theta = updated_params - prev_params
        set_flat_params_to(self.actor, updated_params)

        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

        self.grad_norm_sq_cum = self.grad_norm_sq_cum + torch.norm(grad) ** 2
        return value_loss.item(), action_loss.item(), dist_entropy.item(), grad, d_theta

    # Added by Zebang
    # BEGIN
    def compute_gradient(self, rollouts):
        """

        :param rollouts:
        :return:
        grad: gradient at the current iterate, in a flatten form
        """
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
        action_loss = -(advantages.detach() * magic_box).mean()
        grad = torch.autograd.grad(action_loss, self.actor.parameters(), allow_unused=True, retain_graph=True)
        grad = flatten(grad)

        return grad

    # END

    def inner_update(self, rollouts, prev_grad, d_theta):
        # Added by Zebang
        # BEGIN
        current_grad = self.compute_gradient(rollouts)
        # actor_learning_rate = self.actor_learning_rate_initial / self.grad_norm_sq_cum ** (2 / 3)
        # alpha = self.alpha_initial * actor_learning_rate
        # actor_learning_rate = self.actor_learning_rate_initial/self.iteration**(2/3)
        actor_learning_rate = self.actor_learning_rate_initial
        alpha = self.alpha_initial/self.iteration**(2/3)
        # alpha = 1
        if alpha > 1:
            print("alpha is larger than 1, reset alpha=1")
            alpha = 1
        # END

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
        # print(jacob.shape, d_theta.shape)
        product = torch.dot(jacob, d_theta)
        d_grad = torch.autograd.grad(product, self.actor.parameters(), retain_graph=True)
        grad = (1 - alpha) * prev_grad + alpha * current_grad + (1 - alpha) * flatten(d_grad)

        # update params
        prev_params = get_flat_params_from(self.actor)
        # Added by Zebang
        # BEGIN
        direction = grad/torch.norm(grad)
        # direction = grad
        # END
        updated_params = prev_params - actor_learning_rate * direction
        d_theta = updated_params - prev_params
        set_flat_params_to(self.actor, updated_params)

        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

        self.grad_norm_sq_cum = self.grad_norm_sq_cum + torch.norm(grad) ** 2

        return value_loss.item(), action_loss.item(), dist_entropy.item(), grad, d_theta
