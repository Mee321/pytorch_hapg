import os
import time
from collections import deque
import argparse
from tensorboardX import SummaryWriter

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr.algo.vpg_dice import VPG_DICE
from a2c_ppo_acktr.utils import *
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

GAMMA = 0.995
LR = 3e-4
BATCH_SIZE = 32
NUM_EPOC = 10
BETA = 0.2
TARGET = 0.01
SEED = 1
CUDA = True
ENV_NAME = "Hopper-v2"

# logdir = "./PPO_Penalty/%s/batchsize_%d_%d"%(str(ENV_NAME), BATCH_SIZE, SEED)
# writer = SummaryWriter(log_dir=logdir)
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



agent = VPG_DICE(
        actor_critic,
        0.5,
        0.0,
        lr=0.001)

rollouts = RolloutStorage(5000, 1,
                          envs.observation_space.shape, envs.action_space,
                          actor_critic.recurrent_hidden_state_size)

obs = envs.reset()
rollouts.obs[0].copy_(obs)
rollouts.to(device)

episode_rewards = deque(maxlen=10)
start = time.time()

num_updates = int(
    1e6) // 5000 // 1

for j in range(100):
    # sample
    for step in range(5000):
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
    value_loss, action_loss, dist_entropy = agent.update(rollouts)
    rollouts.after_update()

    if j % 1 == 0 and len(episode_rewards) > 1:
        total_num_steps = (j + 1) * 1 * 5000
        end = time.time()
        print(
            "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
            .format(j, total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards), np.mean(episode_rewards),
                    np.median(episode_rewards), np.min(episode_rewards),
                    np.max(episode_rewards), dist_entropy, value_loss,
                    action_loss))
        # writer.add_scalar("Avg_return", np.mean(episode_rewards), j)