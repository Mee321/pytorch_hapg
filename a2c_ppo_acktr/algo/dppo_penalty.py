import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from a2c_ppo_acktr.utils import *
from torch.autograd import Variable

class PPO_Penalty():
    def __init__(self,
                 actor_critic,
                 kl_coef,
                 kl_target,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef=0.5,
                 entropy_coef=0.0,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=False):

        self.actor_critic = actor_critic
        self.kl_coef = kl_coef
        self.kl_target = kl_target
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        global_net_params = get_flat_params_from(self.actor_critic)
        for e in range(self.ppo_epoch):
            self.kl_coef = 0.2
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample
                # compute means and stds of the old theta
                cur_params = get_flat_params_from(self.actor_critic)
                set_flat_params_to(self.actor_critic, global_net_params)
                _, mean0, std0, _, _, _ = self.actor_critic.evaluate_actions_kl(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)
                set_flat_params_to(self.actor_critic, cur_params)
                # Reshape to do in a single forward pass for all steps
                values, mean1, std1, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions_kl(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)
                print(action_log_probs, old_action_log_probs_batch)
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                mean0 = Variable(mean0.data)
                log_std0 = torch.log(std0)
                log_std0 = Variable(log_std0.data)
                std0 = Variable(std0.data)
                log_std1 = torch.log(std1)
                kls = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
                surr1 = ratio * adv_targ
                #surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                 #                   1.0 + self.clip_param) * adv_targ
                kl_mean = kls.sum(1, keepdim=True).mean()
                action_loss = -surr1.mean() + kl_mean * self.kl_coef
                #print(kl_mean.data, self.kl_coef)
                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                #nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                 #                        self.max_grad_norm)
                self.optimizer.step()

                if kl_mean.data < self.kl_target/1.5:
                    self.kl_coef /= 2
                elif kl_mean.data > self.kl_target * 1.5:
                    self.kl_coef *= 2

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
        updated_params = get_flat_params_from(self.actor_critic)
        set_flat_params_to(self.actor_critic, global_net_params)
        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return updated_params, value_loss_epoch, action_loss_epoch, dist_entropy_epoch
