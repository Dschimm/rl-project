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
        self.net = nn.Sequential(
            nn.Linear(np.prod(env.observation_space.shape), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, env.action_space.shape[0])
        )
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)

    def forward(self, x):
        return self.net(x)

    def update(self, states, actions, rewards, dones, next_states):
        self.optimizer.zero_grad()
        states = torch.tensor(states).float().to(self.device)
        actions = torch.tensor(actions).unsqueeze(-1).to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        dones = torch.tensor(dones).float().to(self.device)
        next_states = torch.tensor(next_states).float().to(self.device)

        q_values = torch.gather(self.net(states), dim=-1,
                                index=actions).squeeze().to(self.device)

        target_q_values = rewards + \
            (1 - dones) * DISCOUNT_FACTOR * \
            self.net(next_states).max(dim=-1)[0].detach()
        loss = F.mse_loss(q_values, target_q_values)

        loss.backward()
        optimizer.step()
        return loss.item()
