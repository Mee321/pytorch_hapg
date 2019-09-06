import torch
import torch.nn as nn
import torch.optim as optim

class VPG_DICE():
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

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(
            actor_critic.parameters(), lr=3e-4)

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

        probs = torch.exp(action_log_probs)
        acc = probs[0]
        prob_acc = torch.autograd.Variable(torch.zeros_like(action_log_probs))
        for i in range(len(probs)):
            if i != 0:
                acc = acc * probs[i] if rollouts.masks[i - 1] != 0 else probs[i]
                print(acc.data, probs[i].data)
            prob_acc[i] = acc

        # acc = action_log_probs[0]
        # prob_acc = torch.autograd.Variable(torch.zeros_like(action_log_probs))
        # for i in range(len(action_log_probs)):
        #     if i != 0:
        #         acc = acc + action_log_probs[i] if rollouts.masks[i - 1] != 0 else action_log_probs[i]
        #     prob_acc[i] = acc
        # prob_acc = torch.exp(prob_acc)
        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()
        # advantages = (advantages - advantages.mean()) / (
        #         advantages.std() + 1e-5)

        probs = torch.exp(action_log_probs)
        magic_box = probs/probs.detach()
        # action_loss = -(rollouts.returns[:-1] * action_log_probs).mean()
        print(torch.min(prob_acc.data))
        action_loss = -(prob_acc/prob_acc.detach() * rewards).mean()
        # action_loss = -(advantages * action_log_probs).mean()
        # print(torch.autograd.grad(action_loss, self.actor_critic.parameters(), allow_unused=True, retain_graph=True))

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()