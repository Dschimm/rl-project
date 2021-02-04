from collections import deque
import random


class ReplayBuffer():

    def __init__(self, seed, batch_size=64, size=1e5):
        self.queue = deque(maxlen=int(size))
        self.batch_size = batch_size
        random.seed(seed)

    def append(self, experience):
        self.queue.append(experience)

    def get_sample(self):
        return random.choices(self.queue, k=self.batch_size)

    def __len__(self):
        return len(self.queue)        

class PrioritizedReplayBuffer():

    def __init__(self, batch_size=64, size=1e5):
        self.queue = deque(maxlen=int(size))
        self.batch_size = batch_size

    def append(self, experience):
        self.queue.append(experience)

    def get_sample(self):
        return random.choices(self.queue, k=self.batch_size)

    def __len__(self):
        return len(self.queue)  