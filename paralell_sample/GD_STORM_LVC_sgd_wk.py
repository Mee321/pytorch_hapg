import os
import time
from collections import deque
from itertools import count
import argparse
from tensorboardX import SummaryWriter

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from hapg.algo.storm_lvc import STORM_LVC
from hapg.utils import *
from hapg.algo import gail
from hapg.arguments import get_args
from hapg.envs import make_vec_envs
from hapg.model import Policy
from hapg.storage import RolloutStorage

GAMMA = 0.995
LR_CRITIC = 3e-2
LR_ACTOR_INITIAL = 3e-2
ALPHA_INITIAL = 1
ALPHA_EXP = 2 / 3
SEED = 41
CUDA = True
ENV_NAME = "Walker2d-v2"
outer_batch = 10000
inner_batch = 10000
num_inner = 0
num_process = 5


logdir = "./GD_SGD/%s/batchsize%d_innersize%d_seed%d_lrcritic%f_lractorinit%f_" % (
    str(ENV_NAME), outer_batch, inner_batch, SEED, LR_CRITIC, LR_ACTOR_INITIAL)
writer = SummaryWriter(log_dir=logdir)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.set_num_threads(num_process)
device = torch.device("cuda:0" if CUDA else "cpu")

envs = make_vec_envs(ENV_NAME, SEED, num_process,
                     GAMMA, "./env_log", device, True)

actor_critic = Policy(envs.observation_space.shape, envs.action_space, base_kwargs={'recurrent': False})
actor_critic.to(device)

agent = STORM_LVC(
    actor_critic=actor_critic,
    value_loss_coef=0.5,
    entropy_coef=0.0,
    critic_learning_rate=LR_CRITIC,
    actor_learning_rate_initial=LR_ACTOR_INITIAL,
    alpha_initial=ALPHA_INITIAL
)

rollouts = RolloutStorage(outer_batch, num_process,
                          envs.observation_space.shape, envs.action_space,
                          actor_critic.recurrent_hidden_state_size)
rollouts_inner = RolloutStorage(inner_batch, num_process,
                                envs.observation_space.shape, envs.action_space,
                                actor_critic.recurrent_hidden_state_size)
obs = envs.reset()
rollouts.obs[0].copy_(obs)
rollouts.to(device)

rollouts_inner.obs[0].copy_(obs)
rollouts_inner.to(device)

episode_rewards = deque(maxlen=10)
total_num_steps = 0

###############################################
#   compute the gradient in the first iteration
###############################################
###############################################
#       Step 1: roll out trajectories
###############################################
for j in count():
    for step in range(outer_batch//num_process):
        # Sample actions
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                rollouts.masks[step])

        # Obser reward and next obs
        obs, reward, done, infos = envs.step(action)
        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])
        # If done then clean the history of observations.
        masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor(
            [[0.0] if 'bad_transition' in info.keys() else [1.0]
             for info in infos])
        rollouts.insert(obs, recurrent_hidden_states, action,
                        action_log_prob, value, reward, masks, bad_masks)

    with torch.no_grad():
        next_value = actor_critic.get_value(
            rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
            rollouts.masks[-1]).detach()

    ###############################################
    #       Step 2: process sample
    ###############################################
    rollouts.compute_returns(next_value, True, 0.99,
                             0.97, True)
    ###############################################
    #       Step 3: update parameters
    ###############################################
    prev_params = get_flat_params_from(actor_critic)
    value_loss, action_loss, dist_entropy, grad, d_theta = agent.update(rollouts)
    cur_params = get_flat_params_from(actor_critic)
    rollouts.after_update()
    total_num_steps += outer_batch
    print(total_num_steps, np.mean(episode_rewards))
    writer.add_scalar("Avg_return", np.mean(episode_rewards), total_num_steps)
    writer.add_scalar("grad_norm", torch.norm(grad), total_num_steps)

    # grad_norm_sq_cum = grad_norm_sq_cum + torch.norm(grad)**2

    for inner_update in range(num_inner):
        agent.iteration = inner_update+2
        ###############################################
        #   compute the gradient in the following iterations
        ###############################################
        ###############################################
        #       Step 1: roll out trajectories
        ###############################################
        a = np.random.uniform()
        #test_critic
        alignment = sum([len(p.view(-1)) for p in list(actor_critic.parameters())[4:10]])
        mix_params = a * prev_params + (1 - a) * cur_params
        mix_params[-alignment:] = cur_params[-alignment:]

        set_flat_params_to(actor_critic, mix_params)
        for step in range(inner_batch):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts_inner.obs[step], rollouts_inner.recurrent_hidden_states[step],
                    rollouts_inner.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
            if done:
                obs = envs.reset()
            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts_inner.insert(obs, recurrent_hidden_states, action,
                                  action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts_inner.obs[-1], rollouts_inner.recurrent_hidden_states[-1],
                rollouts_inner.masks[-1]).detach()

        ###############################################
        #       Step 2: process sample
        ###############################################
        rollouts_inner.compute_returns(next_value, True, 0.99, 0.97, True)
        ###############################################
        #       Step 3: update parameters
        ###############################################
        set_flat_params_to(actor_critic, cur_params)
        prev_params = cur_params
        value_loss, action_loss, dist_entropy, grad, d_theta = agent.inner_update(rollouts_inner, grad, d_theta)
        rollouts_inner.after_update()
        cur_params = get_flat_params_from(actor_critic)

        total_num_steps += inner_batch
        print(total_num_steps, np.mean(episode_rewards))
        writer.add_scalar("Avg_return", np.mean(episode_rewards), total_num_steps)
        writer.add_scalar("grad_norm", torch.norm(grad), total_num_steps)
        if inner_update % 10 == 0 and len(episode_rewards) > 1:
            print(
                "Updates {}, num timesteps {}\n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, "
                "min/max reward {:.1f}/{:.1f}\n "
                    .format(inner_update, total_num_steps,
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))
            print("grad_sq_norm_cum {}\n".format(agent.grad_norm_sq_cum))

    if total_num_steps > 4e6:
        break
