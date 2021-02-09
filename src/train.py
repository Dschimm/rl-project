import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

import gym
from gym.wrappers.resize_observation import ResizeObservation
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.gray_scale_observation import GrayScaleObservation

from buffer import ReplayBuffer, PrioritizedReplayBuffer
from policy import RandomPolicy, eGreedyPolicy
from dqn import DQN
from agent import Agent


from utils import get_latest_model, save_checkpoint, load_checkpoint, get_cuda_device
from gym_utils import ActionWrapper


def train(env, agent, EPISODES=1000, EPISODE_LENGTH=10000, BATCH_SIZE=64):

    rewards = []
    frames = 0
    for i in range(EPISODES):
        state = env.reset()
        print("Episode", i)
        for j in range(EPISODE_LENGTH):
            # env.render()
            print(".", end="", flush=True)
            action = agent.act(state)
            next_state, reward, done, meta = env.step(action)
            frames += 1

            rewards.append(reward)
            state = next_state

            agent.fill_buffer((state, action, reward, done, next_state))

            if frames > 80000 and len(agent.buffer) >= BATCH_SIZE:
                agent.update()

            if done:
                break

        if i % 100 == 0:
            print(i)
            print("Mean reward:", np.mean(rewards))
            rewards.clear()

    save_checkpoint(agent.model, "dqn")
    return


if __name__ == "__main__":
    seed = 42
    env = ResizeObservation(GrayScaleObservation(gym.make('CarRacing-v0')), 64)
    env = ActionWrapper(env)
    env = FrameStack(env, 4)
    env.seed(seed)
    dqn = DQN(env)
    policy = eGreedyPolicy(env, seed, 0.1, dqn)
    buffer = PrioritizedReplayBuffer(seed)
    agent = Agent(dqn, policy, buffer)
    train(env, agent, EPISODES=10)
