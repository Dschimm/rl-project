import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

from utils import get_cuda_device


class DQN(nn.Module):

    def __init__(self, env):
        super(DQN, self).__init__()
        self.device = get_cuda_device()
        # env.observation space (0, 255, (96, 96, 3), uint8)
        # with atari preprocessing
        
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=16) #81x81x6
        self.pool = nn.MaxPool2d(2,2) #40x40x6
        self.c2 = nn.Conv2d(in_channels=6, out_channels=4, kernel_size=8) #33x33x4
        self.fc1 = nn.Linear(16*16*4, 64)
        self.fc2 = nn.Linear(64, env.action_space.shape[0])

        self.relu = nn.ReLU()
        self.to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        # [batch_size, channel_size, height, width]
        x = x.unsqueeze(1) # add channel size 1 for greyscaling
        y = self.pool(self.relu(self.c1(x)))
        y = self.pool(self.relu(self.c2(y)))
        y = y.view(-1, 16*16*4)
        y = self.relu(self.fc1(y))
        y = self.fc2(y)
        return y

    def update(self, states, actions, rewards, dones, next_states):
        self.optimizer.zero_grad()
        states = torch.tensor(states).float().to(self.device)
        actions = torch.tensor(actions).unsqueeze(-1).to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        dones = torch.tensor(dones).float().to(self.device)
        next_states = torch.tensor(next_states).float().to(self.device)

        q_values = torch.gather(self(states), dim=-1,
                                index=actions).squeeze().to(self.device)
    # FIX LEARNING AND OUTPUT OF NET
        target_q_values = rewards + \
            (1 - dones) * DISCOUNT_FACTOR * \
            self(next_states).max(dim=-1)[0].detach()
        loss = F.mse_loss(q_values, target_q_values)

        loss.backward()
        optimizer.step()
        return loss.item()
