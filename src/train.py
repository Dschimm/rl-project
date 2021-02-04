import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

import gym
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from collections import deque

from policy import RandomPolicy
from dqn import DQN


from utils import get_latest_model, save_checkpoint, load_checkpoint, get_cuda_device
from gym_utils import ActionWrapper



def train(env, model, policy, EPISODES=1000, EPISODE_LENGTH = 100, BATCH_SIZE = 64):
    
    rewards = []
    replay_buffer = deque(maxlen=int(1e5))
    
    for i in range(EPISODES):
        state = env.reset()
        print("Episode", i)
        for _ in range(EPISODE_LENGTH):    
            action = policy.action(state)
            next_state, reward, done, meta = env.step(action)

            rewards.append(reward)
            state = next_state

            replay_buffer.append((state, action, reward, done, next_state))
            
            if len(replay_buffer) >= BATCH_SIZE:
                batch = random.choices(replay_buffer, k=BATCH_SIZE)
                model.update(*zip(*batch))
            
            if done:
                break

        if i % 100 == 0:
            print(i)
            print("Mean reward:", np.mean(rewards))
            rewards.clear()

    save_checkpoint(Q,"dqn")
    return

if __name__ == "__main__":
    env = ActionWrapper(GrayScaleObservation(gym.make('CarRacing-v0')))
    env.seed(42)
    dqn = DQN(env)
    policy = RandomPolicy(42)
    train(env, dqn, policy)