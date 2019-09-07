import torch
import torch.nn as nn
import torch.optim as optim
from hapg.utils import *

# LVC version, DiCE with a bug when denominator becomes 0
class HAPG_LVC():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 lr_inner=None,
                 max_grad_norm=None):

        self.actor_critic = actor_critic
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.alignment = sum([len(p.view(-1)) for p in list(self.actor_critic.parameters())[4:10]])
        # value net optimizer
        self.optimizer = optim.Adam(
            actor_critic.parameters(), lr=self.lr)

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        rewards = rollouts.rewards
        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        # probs = torch.exp(action_log_probs)
        # acc = probs[0]
        # prob_acc = torch.autograd.Variable(torch.zeros_like(action_log_probs))
        # for i in range(len(probs)):
        #     if i != 0:
        #         acc = acc * probs[i] if rollouts.masks[i - 1] != 0 else probs[i]
        #     prob_acc[i] = acc

        # acc = action_log_probs[0]
        # prob_acc = torch.autograd.Variable(torch.zeros_like(action_log_probs))
        # for i in range(len(action_log_probs)):
        #     if i != 0:
        #         acc = acc + action_log_probs[i] if rollouts.masks[i - 1] != 0 else action_log_probs[i]
        #     prob_acc[i] = acc
        # prob_acc = torch.exp(prob_acc)
        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)
        probs = torch.exp(action_log_probs)

        magic_box = probs/probs.detach()
        # action_loss = -(rollouts.returns[:-1] * action_log_probs).mean()
        # action_loss = -(prob_acc * rewards).mean()
        action_loss = -(advantages.detach() * magic_box).mean()
        # print(torch.autograd.grad(action_loss, self.actor_critic.parameters(), allow_unused=True))
        grad = torch.autograd.grad(action_loss, self.actor_critic.parameters(), allow_unused=True, retain_graph=True)
        grad = flatten_tuple(grad, self.alignment)

        prev_params = get_flat_params_from(self.actor_critic)
        direction = grad / torch.norm(grad)
        updated_params = prev_params - self.lr * direction
        d_theta = updated_params - prev_params
        set_flat_params_to(self.actor_critic, updated_params)

        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item(), grad, d_theta

    def inner_update(self, rollouts, prev_grad, d_theta):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        rewards = rollouts.rewards
        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        # probs = torch.exp(action_log_probs)
        # acc = probs[0]
        # prob_acc = torch.autograd.Variable(torch.zeros_like(action_log_probs))
        # for i in range(len(probs)):
        #     if i != 0:
        #         acc = acc * probs[i] if rollouts.masks[i - 1] != 0 else probs[i]
        #     prob_acc[i] = acc

        acc = action_log_probs[0]
        prob_acc = torch.autograd.Variable(torch.zeros_like(action_log_probs))
        for i in range(len(action_log_probs)):
            if i != 0:
                acc = acc + action_log_probs[i] if rollouts.masks[i - 1] != 0 else action_log_probs[i]
            prob_acc[i] = acc
        prob_acc = torch.exp(prob_acc)
        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)
        probs = torch.exp(action_log_probs)

        magic_box = probs / probs.detach()
        # action_loss = -(rollouts.returns[:-1] * action_log_probs).mean()
        # action_loss = -(magic_box * rewards).mean()
        # action_loss = -(prob_acc * rewards).mean()
        action_loss = -(magic_box * advantages.detach()).mean()
        jacob = torch.autograd.grad(action_loss, self.actor_critic.parameters(), allow_unused=True, retain_graph=True, create_graph=True)
        jacob = flatten_tuple(jacob, self.alignment)
        # print(jacob.shape, d_theta.shape)
        product = torch.dot(jacob, d_theta)
        d_grad = torch.autograd.grad(product, self.actor_critic.parameters(), allow_unused=True, retain_graph=True)
        grad = prev_grad + flatten_tuple(d_grad, self.alignment)

        # update params
        prev_params = get_flat_params_from(self.actor_critic)
        direction = grad / torch.norm(grad)
        updated_params = prev_params - self.lr * direction
        d_theta = updated_params - prev_params
        set_flat_params_to(self.actor_critic, updated_params)

        # action_loss = -(advantages * action_log_probs).mean()
        # print(torch.autograd.grad(action_loss, self.actor_critic.parameters(), allow_unused=True))

        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item(), grad, d_theta
