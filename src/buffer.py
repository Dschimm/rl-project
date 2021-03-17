from collections import deque
import random
import numpy as np
import config as cfg


class ReplayBuffer:
    """A simple Replay Buffer"""
    def __init__(self, seed, batch_size=cfg.BATCH_SIZE, size=cfg.BUFFER_SIZE):
        self.queue = deque(maxlen=int(size))
        self.batch_size = batch_size
        random.seed(seed)

    def append(self, experience):
        """Appends experience to the buffer"""
        self.queue.append(experience)

    def get_sample(self):
        """Randomly sample from buffer according to batch size"""
        return random.choices(self.queue, k=self.batch_size)

    def update_weights(self, targets):
        """No effect for simple buffer"""
        return

    def __len__(self):
        return len(self.queue)


class PrioritizedReplayBuffer:
    """Prioritized Replay Buffer that samples according to weights."""
    def __init__(self, batch_size=cfg.BATCH_SIZE, size=cfg.BUFFER_SIZE):
        self.max_size = size
        self.experience = list()
        self.prios = list()
        self.probs = list()
        self.batch_size = batch_size

    def append(self, sample):
        """Adds a sample to the buffer and assigns an initially high weight."""
        if len(self.experience) > self.max_size:
            index = np.argmin(self.prios)
            self.experience[index] = sample
            self.prios[index] = 100000000

        else:
            self.experience.append(sample)
            self.prios.append(100000000)

        self.compute_probs()

    def get_sample(self):
        """Sample according to priority"""
        self.last_indices = np.random.choice(
            list(range(len(self.experience))),
            size=self.batch_size,
            replace=False,
            p=self.probs,
        )
        return [self.experience[i] for i in self.last_indices]

    def update_weights(self, targets):
        """Update weights of the experience tuples according to loss"""
        for i, t in enumerate(targets):
            self.prios[self.last_indices[i]] = t

        self.compute_probs()

    def compute_probs(self):
        """Normalize weights"""
        p_sum = sum(self.prios)
        self.probs = [p / p_sum for p in self.prios]

    def __len__(self):
        return len(self.experience)
