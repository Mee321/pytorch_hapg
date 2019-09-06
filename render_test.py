import gym
import numpy as np

env = gym.make("Pong-v4")
env.reset()
action_space = env.action_space
for _ in range(10000):
    env.render()
    env.step(env.action_space.sample())
env.close()