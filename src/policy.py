import numpy as np
import torch


class RandomPolicy:
    def __init__(self, seed=None):
        np.random.seed(seed)

    def action(self, state):
        return np.random.randint(9)


class NoExplorationPolicy:
    def __init__(self, model):
        self.model = model

    def action(self, state):
        state = torch.tensor(state).unsqueeze(0).float()
        q_values = self.model(state)
        return torch.max(q_values, dim=1)[1].item()


class eGreedyPolicy:
    def __init__(self, env, seed, eps, model):
        self.eps = eps
        np.random.seed(seed)
        self.env = env
        self.model = model

    def action(self, state):
        if np.random.uniform(0, 1) < self.eps:
            return self.env.action_space.sample()
        state = torch.tensor(state).unsqueeze(0).float()
        q_values = self.model(state)
        return torch.max(q_values, dim=1)[1].item()
