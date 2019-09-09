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


from hapg.algo.seperated_storm import STORM_LVC
from hapg.utils import *
from hapg.algo import gail
from hapg.arguments import get_args
from hapg.envs import make_vec_envs
from hapg.model import Policy
from hapg.seperated_model_consistent import *
from hapg.model import Policy
from hapg.storage import RolloutStorage

GAMMA = 0.995
ACTOR_LR = 0.03
CRITIC_LR = 0.03
SEED = 1
CUDA = True
ENV_NAME = "HalfCheetah-v2"
outer_batch = 1000
inner_batch = 1000
num_inner = 10


logdir = "./STORM_LVC/%s/batchsize%d_innersize%d_seed%d_lr%f" % (
str(ENV_NAME), outer_batch, inner_batch, SEED, ACTOR_LR)
writer = SummaryWriter(log_dir=logdir)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.set_num_threads(1)
device = torch.device("cuda:0" if CUDA else "cpu")

envs = make_vec_envs(ENV_NAME, SEED, 1,
                     GAMMA, "./", device, False)

actor_critic = Policy(envs.observation_space.shape, envs.action_space, base_kwargs={'recurrent': False})
actor = Actor(envs.observation_space.shape[0], envs.action_space.shape[0], hidden_size=64)
critic = Critic(envs.observation_space.shape[0], hidden_size=64)

# copy ac to a and c
actor_params_1 = get_flat_params_from(actor_critic.base.actor)
actor_params_2 = get_flat_params_from(actor_critic.dist)
critic_params_1 = get_flat_params_from(actor_critic.base.critic)
critic_params_2 = get_flat_params_from(actor_critic.base.critic_linear)
actor_params = torch.cat([actor_params_1, actor_params_2])
critic_params = torch.cat([critic_params_1, critic_params_2])
set_flat_params_to(actor, actor_params)
set_flat_params_to(critic, critic_params)

actor.to(device)
critic.to(device)

agent = STORM_LVC(
    actor,
    critic,
    0.5,
    0.0,
    critic_learning_rate=CRITIC_LR,
    actor_learning_rate_initial=ACTOR_LR)

rollouts = RolloutStorage(outer_batch, 1,
                          envs.observation_space.shape, envs.action_space,
                          1)
rollouts_inner = RolloutStorage(inner_batch, 1,
                                envs.observation_space.shape, envs.action_space,
                                1)

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
            value = critic(rollouts.obs[step])
            action, action_log_prob, dist_entropy = actor.act(rollouts.obs[step])
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
        rollouts.insert(obs, torch.tensor(0.0), action,
                        action_log_prob, value, reward, masks, bad_masks)

    with torch.no_grad():
        next_value = critic.get_value(rollouts.obs[-1]).detach()

    # process sample
    rollouts.compute_returns(next_value, True, 0.99,
                             0.97, True)

    # compute updated params
    prev_params = get_flat_params_from(actor)
    value_loss, action_loss, dist_entropy, grad, d_theta = agent.update(rollouts)
    cur_params = get_flat_params_from(actor)
    rollouts.after_update()
    total_num_steps += outer_batch
    print(total_num_steps, np.mean(episode_rewards))
    writer.add_scalar("Avg_return", np.mean(episode_rewards), total_num_steps)
    writer.add_scalar("grad_norm", torch.norm(grad), total_num_steps)

    for inner_update in range(num_inner):
        a = np.random.uniform()
        mix_params = a * prev_params + (1 - a) * cur_params
        set_flat_params_to(actor, mix_params)
        for step in range(inner_batch):
            # Sample actions
            with torch.no_grad():
                value = critic(rollouts.obs[step])
                action, action_log_prob, dist_entropy = actor.act(rollouts.obs[step])

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
            rollouts_inner.insert(obs, torch.tensor(0.0), action,
                                  action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = critic.get_value(rollouts.obs[-1]).detach()
        # process sample
        rollouts_inner.compute_returns(next_value, True, 0.99,
                                       0.97, True)
        # compute updated params
        set_flat_params_to(actor, cur_params)
        prev_params = cur_params
        value_loss, action_loss, dist_entropy, grad, d_theta = agent.inner_update(rollouts_inner, grad, d_theta)
        rollouts_inner.after_update()
        cur_params = get_flat_params_from(actor)

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

    if total_num_steps > 3e6:
        break