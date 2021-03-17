import numpy as np
import torch


class RandomPolicy:
    """
    Get a random action.
    """

    def __init__(self, seed=None):
        np.random.seed(seed)

    def action(self, state):
        return np.random.randint(9)


class NoExplorationPolicy:
    """
    Get the best action according to the policy.
    """

    def __init__(self, model):
        self.model = model

    def action(self, state):
        state = torch.tensor(state).unsqueeze(0).float().to(self.model.device)
        q_values = self.model(state)
        return torch.max(q_values, dim=1)[1].item()


class eGreedyPolicy:
    """
    Standard epsilon greedy policy. The policy chooses the best action with probability
    1-eps and a random action with probability eps.
    """

    def __init__(self, env, seed, eps, model):
        self.eps = eps
        np.random.seed(seed)
        self.env = env
        self.model = model

    def action(self, state):
        if np.random.uniform(0, 1) < self.eps:
            return self.env.action_space.sample()
        state = torch.tensor(state).unsqueeze(0).float().to(self.model.device)
        q_values = self.model(state)
        return torch.max(q_values, dim=1)[1].item()


class eGreedyPolicyDecay(eGreedyPolicy):
    """
    Extension of the epsilon greedy policy with a linear decay over a set amount of steps.
    """

    def __init__(self, env, seed, eps, eps_start, eps_end, decay_steps, model):
        super(eGreedyPolicyDecay, self).__init__(env, seed, eps, model)
        self.decay = (eps_start - eps_end) / decay_steps
        self.eps_end = eps_end

    def decay_eps(self):
        """
        Update epsilon with the decay.
        """
        if self.eps > self.eps_end:
            self.eps -= self.decay
