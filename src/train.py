import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

import pickle

import gym

from buffer import ReplayBuffer, PrioritizedReplayBuffer
from policy import RandomPolicy, eGreedyPolicy
from dqn import DQN
from agent import Agent, DDQNAgent

from utils import get_latest_model, save_checkpoint, load_checkpoint, get_cuda_device
from gym_utils import getWrappedEnv


def train(env, agent, EPISODES=10000, EPISODE_LENGTH=10000, SKIP_FRAMES=80000, BATCH_SIZE=64):

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

            if frames > SKIP_FRAMES and len(agent.buffer) >= BATCH_SIZE:
                with open("../models/buffer.pkl", "wb+") as f:
                    pickle.dump(agent.buffer, f)
                agent.update()

            if done:
                break

        if i % 100 == 0:
            print(i)
            print("Mean reward:", np.mean(rewards))
            rewards.clear()

    save_checkpoint(agent.model, "dqn")
    print(frames)
    return


if __name__ == "__main__":
    seed = 42
    env = getWrappedEnv(seed=seed)
    dqn = DQN(env)
    eval_net = DQN(env)
    policy = eGreedyPolicy(env, seed, 0.1, dqn)
    buffer = PrioritizedReplayBuffer(seed)
    agent = DDQNAgent(dqn, eval_net, policy, buffer)
    train(env, agent)
