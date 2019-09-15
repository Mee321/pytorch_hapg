import torch
import torch.nn as nn
import torch.optim as optim
from hapg.utils import *
from hapg.linear_optimization_oracles import LONuclearNormBall

# LVC version, DiCE with a bug when denominator becomes 0
class FWSCG():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 critic_learning_rate=None,
                 actor_learning_rate_initial=1,
                 alpha_initial=1,
                 max_grad_norm=None,
                 device="cpu"):

        self.actor_critic = actor_critic
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate_initial = actor_learning_rate_initial
        self.alpha_initial = alpha_initial
        self.max_grad_norm = max_grad_norm
        self.alignment = sum([len(p.view(-1)) for p in list(self.actor_critic.parameters())[4:10]])
        # value net optimizer
        self.optimizer = optim.Adam(
            actor_critic.parameters(), lr=self.critic_learning_rate)
        self.grad_norm_sq_cum = 0
        self.iteration = 1
        self.device = device
        self.LO = LONuclearNormBall(1e1)

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

        magic_box = probs / probs.detach()
        # action_loss = -(rollouts.returns[:-1] * action_log_probs).mean()
        # action_loss = -(prob_acc * rewards).mean()
        action_loss = -(advantages.detach() * magic_box).mean()
        # print(torch.autograd.grad(action_loss, self.actor_critic.parameters(), allow_unused=True))
        grad = torch.autograd.grad(action_loss, self.actor_critic.parameters(), allow_unused=True, retain_graph=True)
        grad = flatten_tuple(grad, self.alignment, self.device)

        prev_params = get_flat_params_from(self.actor_critic)

        direction = self.get_update_direction_with_lo(grad, self.actor_critic)
        # print(torch.norm(direction-grad))

        # direction = grad / torch.norm(grad)
        direction = direction/torch.norm(direction)
        # grad_normalize = grad/torch.norm(grad)
        #
        # print(torch.norm(direction - grad_normalize))

        updated_params = prev_params - self.actor_learning_rate_initial * direction
        d_theta = updated_params - prev_params
        set_flat_params_to(self.actor_critic, updated_params)

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

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
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
        grad = torch.autograd.grad(action_loss, self.actor_critic.parameters(), allow_unused=True, retain_graph=True)
        grad = flatten_tuple(grad, self.alignment, self.device)

        # self.optimizer.zero_grad()
        # value_loss.backward()
        # self.optimizer.step()

        return grad

    # END

    def inner_update(self, rollouts, prev_grad, d_theta, current_grad):
        # Added by Zebang
        # BEGIN
        # current_grad = self.compute_gradient(rollouts)
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

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

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
        action_loss = -(magic_box * advantages.detach()).mean()
        jacob = torch.autograd.grad(action_loss, self.actor_critic.parameters(), allow_unused=True, retain_graph=True,
                                    create_graph=True)
        jacob = flatten_tuple(jacob, self.alignment, self.device)
        product = torch.dot(jacob, d_theta)
        d_grad = torch.autograd.grad(product, self.actor_critic.parameters(), allow_unused=True, retain_graph=True)
        # grad = (1 - alpha) * prev_grad + alpha * current_grad + (1 - alpha) * flatten_tuple(d_grad, self.alignment, self.device)
        grad = (1 - alpha) * prev_grad + alpha * current_grad

        # update params
        prev_params = get_flat_params_from(self.actor_critic)
        # Added by Zebang
        # BEGIN
        direction = self.get_update_direction_with_lo(grad, self.actor_critic)
        # print(torch.norm(direction-grad))


        # direction = grad / torch.norm(grad)
        direction = direction / torch.norm(direction)
        # grad_normalize = grad / torch.norm(grad)
        # print(torch.norm(direction - grad_normalize))
        # END
        updated_params = prev_params - actor_learning_rate * direction
        d_theta = updated_params - prev_params
        set_flat_params_to(self.actor_critic, updated_params)

        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

        # DEBUG #
        # params_after_adam_step = get_flat_params_from(self.actor_critic)
        #
        # prev_ind = 0
        # params_difference_after_adam_step = params_after_adam_step - updated_params
        # for param in self.actor_critic.parameters():
        #     flat_size = int(np.prod(list(param.size())))
        #     print(torch.norm(params_difference_after_adam_step[prev_ind:prev_ind + flat_size]))
        #     prev_ind += flat_size

        # END DEBUG #

        self.grad_norm_sq_cum = self.grad_norm_sq_cum + torch.norm(grad) ** 2

        return value_loss.item(), action_loss.item(), dist_entropy.item(), grad, d_theta

    def get_update_direction_with_lo(self, grad_flat, current_net):
        # if self.debug:
            # print("This is DEBUG")
            # return -grad_flat
        directions = []
        prev_ind = 0
        block = 0
        for param in current_net.parameters():
            flat_size = int(np.prod(list(param.size())))

            # ndarray = grad_flat[prev_ind:prev_ind + flat_size].view(param.size()).detach().numpy()
            ndarray = grad_flat[prev_ind:prev_ind + flat_size].view(param.size()).detach()
            # print(ndarray.dim())
            if block < 4 or block > 9:  # actor block
                if ndarray.dim() > 1:  # inter-layer parameters
                    ndarray = ndarray.numpy()
                    direction_layer = self.LO.lo_oracle(-ndarray)
                    direction_layer = torch.from_numpy(direction_layer).view(-1)
                    direction_layer = param.view(-1) - direction_layer.float()
                    #########DEBUG###########
                    # direction_layer = ndarray.view(-1)
                    #########################
                    # if self.iteration % 100 == 0:
                    #     print(torch.norm(param.view(-1)) / torch.norm(direction_layer))
                else:  # parameters of activation functions
                    # direction_layer = torch.from_numpy(-ndarray)
                    direction_layer = -ndarray
            else:  # critic block
                direction_layer = ndarray.view(-1)
                # print(torch.norm(direction_layer))
            block += 1
            directions.append(direction_layer)
            prev_ind += flat_size
        direction = torch.cat(directions)
        # print(torch.norm(-grad_flat - direction))
        return direction
