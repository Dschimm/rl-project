from collections import deque
import random
import numpy as np


class ReplayBuffer():

    def __init__(self, seed, batch_size=64, size=1e5):
        self.queue = deque(maxlen=int(size))
        self.batch_size = batch_size
        random.seed(seed)

    def append(self, experience):
        self.queue.append(experience)

    def get_sample(self):
        return random.choices(self.queue, k=self.batch_size)
    
    def update_weights(self, targets):
        return

    def __len__(self):
        return len(self.queue)        

class PrioritizedReplayBuffer():

    def __init__(self, batch_size=64, size=1e5):
        self.max_size = size
        self.experience = list()
        self.prios = list() 
        self.probs = list()
        self.batch_size = batch_size

    def append(self, sample):

        if len(self.experience) > self.max_size:
            index = np.argmin(self.prios)
            self.experience[index] = sample
            self.prios[index] = 100000000

        else:
            self.experience.append(sample)
            self.prios.append(100000000)

        self.compute_probs()

    def get_sample(self):
        # sample according to priority
        self.last_indices = random.choices([range(len(self.experience))], k=self.batch_size, weights=self.probs) 
        return [self.experience[i] for i in self.last_indices]

    def update_weights(self, targets):
        for i, t in enumerate(targets):
            self.prios[self.last_indices[i]] = t
        
        self.compute_probs()

    def compute_probs(self):
        p_sum = sum(self.prios)
        self.probs = [p/p_sum for p in self.prios]

    def __len__(self):
        return len(self.queue)  