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

from a2c_ppo_acktr.algo.hapg_dice import HAPG_DICE
from a2c_ppo_acktr.utils import *
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

GAMMA = 0.995
LR = 3e-3
BATCH_SIZE = 32
NUM_EPOC = 10
BETA = 0.2
TARGET = 0.01
SEED = 1
CUDA = True
ENV_NAME = "Walker2d-v2"
outer_batch = 10000
inner_batch = 1000
num_inner = 10

for SEED in [11, 21]:
    for ENV_NAME in ["HalfCheetah-v2", "Walker2d-v2", "Hopper-v2", "Humanoid-v2"]:
        logdir = "./HAPG_LVC/%s/batchsize%d_innersize%d_seed%d_lr%f"%(str(ENV_NAME),outer_batch, inner_batch, SEED, LR)
        writer = SummaryWriter(log_dir=logdir)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        torch.set_num_threads(1)
        device = torch.device("cuda:0" if CUDA else "cpu")

        envs = make_vec_envs(ENV_NAME, SEED, 1,
                         GAMMA, "./", device, False)

        actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': False})
        actor_critic.to(device)



        agent = HAPG_DICE(
            actor_critic,
            0.5,
            0.0,
            lr=LR)

        rollouts = RolloutStorage(outer_batch, 1,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
        rollouts_inner = RolloutStorage(inner_batch, 1,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

        obs = envs.reset()
        rollouts.obs[0].copy_(obs)
        rollouts.to(device)

        rollouts_inner.obs[0].copy_(obs)
        rollouts_inner.to(device)

        episode_rewards = deque(maxlen=10)
        total_num_steps = 0

        for j in count():
        # sample
            for step in range(outer_batch):
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

            # process sample
            rollouts.compute_returns(next_value, True, 0.99,
                                     0.97, True)

            # compute updated params
            prev_params = get_flat_params_from(actor_critic)
            value_loss, action_loss, dist_entropy, grad, d_theta = agent.update(rollouts)
            cur_params = get_flat_params_from(actor_critic)
            rollouts.after_update()
            total_num_steps += outer_batch
            writer.add_scalar("Avg_return", np.mean(episode_rewards), total_num_steps)
            writer.add_scalar("grad_norm", torch.norm(grad), total_num_steps)

            for inner_update in range(num_inner):
                a = np.random.uniform()
                mix_params = a*prev_params + (1-a)*cur_params
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

                # process sample
                rollouts_inner.compute_returns(next_value, True, 0.99,
                                         0.97, True)
                # compute updated params
                set_flat_params_to(actor_critic, cur_params)
                prev_params = cur_params
                value_loss, action_loss, dist_entropy, grad, d_theta = agent.inner_update(rollouts_inner, grad, d_theta)
                rollouts_inner.after_update()
                cur_params = get_flat_params_from(actor_critic)

                total_num_steps += inner_batch
                print(total_num_steps, np.mean(episode_rewards))
                writer.add_scalar("Avg_return", np.mean(episode_rewards), total_num_steps)
                writer.add_scalar("grad_norm", torch.norm(grad), total_num_steps)

            if j % 1 == 0 and len(episode_rewards) > 1:
                print(
                    "Updates {}, num timesteps {}\n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps,
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))
            if total_num_steps > 3e6:
                break