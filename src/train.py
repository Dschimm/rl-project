import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

import gym
from gym.wrappers.gray_scale_observation import GrayScaleObservation

from buffer import ReplayBuffer, PrioritizedReplayBuffer
from policy import RandomPolicy, eGreedyPolicy
from dqn import DQN


from utils import get_latest_model, save_checkpoint, load_checkpoint, get_cuda_device
from gym_utils import ActionWrapper



def train(env, model, policy, EPISODES=1000, EPISODE_LENGTH = 200, BATCH_SIZE = 64):
    
    rewards = []
    replay_buffer = ReplayBuffer(42, batch_size=BATCH_SIZE)
    for i in range(EPISODES):
        state = env.reset()
        print("Episode", i)
        for _ in range(EPISODE_LENGTH):  
            #env.render()
            print(".", end="", flush=True)
            action = policy.action(state)
            next_state, reward, done, meta = env.step(action)

            rewards.append(reward)
            state = next_state

            replay_buffer.append((state, action, reward, done, next_state))
            
            if len(replay_buffer) >= BATCH_SIZE:
                batch = replay_buffer.get_sample()
                model.update(*zip(*batch))
            
            if done:
                break

        if i % 100 == 0:
            print(i)
            print("Mean reward:", np.mean(rewards))
            rewards.clear()

    save_checkpoint(model,"dqn")
    return

if __name__ == "__main__":
    seed = 42
    env = ActionWrapper(GrayScaleObservation(gym.make('CarRacing-v0')))
    env.seed(seed)
    dqn = DQN(env)
    policy = eGreedyPolicy(env, seed, 0.1, dqn)
    train(env, dqn, policy, EPISODES=10)