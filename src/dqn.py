import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from utils import get_cuda_device
import config as cfg


class DQN(nn.Module):
    def __init__(self, env, lr=cfg.LEARNING_RATE):
        super(DQN, self).__init__()
        self.device = get_cuda_device()
        self.discount = cfg.DISCOUNT_FACTOR
        self.lr = lr
        # env.observation space (0, 255, (96, 96, 3), uint8)
        # input should be (N, Cin, D, H, W)
        # with atari preprocessing

        self.c1 = nn.Conv3d(in_channels=1, out_channels=6,
                            kernel_size=(2, 16, 16))  # 81x81x6
        self.pool = nn.MaxPool3d(2, stride=2)  # 40x40x6
        self.c2 = nn.Conv2d(in_channels=6, out_channels=4,
                            kernel_size=8)  # 33x33x4
        self.fc1 = nn.Linear(8 * 8 * 2, 64)
        self.fc2 = nn.Linear(64, env.action_space.n)

        self.relu = nn.ReLU()
        self.to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        # [batch_size, channel_size, height, width]
        x = x.unsqueeze(1)  # add channel size 1 for greyscaling
        x = x.squeeze(-1)
        y = self.pool(self.relu(self.c1(x)))
        y = y.squeeze(2)
        y = self.pool(self.relu(self.c2(y)))
        y = y.view(-1, 8 * 8 * 2)
        y = self.relu(self.fc1(y))
        y = self.fc2(y)
        return y


class DuelingDQN(nn.Module):
    # estimates Value function
    def __init__(self, env, lr=cfg.LEARNING_RATE):
        super(DuelingDQN, self).__init__()
        self.device = get_cuda_device()
        self.discount = cfg.DISCOUNT_FACTOR
        self.lr = lr
        # env.observation space (0, 255, (96, 96, 3), uint8)
        # input should be (N, Cin, D, H, W)
        # with atari preprocessing

        self.c1 = nn.Conv3d(in_channels=1, out_channels=6,
                            kernel_size=(2, 16, 16))  # 81x81x6
        self.pool = nn.MaxPool3d(2, stride=2)  # 40x40x6
        self.c2 = nn.Conv2d(in_channels=6, out_channels=4,
                            kernel_size=8)  # 33x33x4
        self.fcA1 = nn.Linear(8 * 8 * 2, 64)
        self.fcV1 = nn.Linear(8 * 8 * 2, 64)
        self.fcA2 = nn.Linear(64, env.action_space.n)
        self.fcV2 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        # [batch_size, channel_size, height, width]
        x = x.unsqueeze(1)  # add channel size 1 for greyscaling
        x = x.squeeze(-1)
        y = self.pool(self.relu(self.c1(x)))
        y = y.squeeze(2)
        y = self.pool(self.relu(self.c2(y)))
        y = y.view(-1, 8 * 8 * 2)
        A = self.relu(self.fcA1(y))
        A = self.relu(self.fcA2(A))
        V = self.relu(self.fcV1(y))
        V = self.relu(self.fcV2(V))
        A = torch.sub(A, torch.mean(A))  # stabilizes learning
        return torch.add(A, V)
