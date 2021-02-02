import numpy as np

class RandomPolicy():
    
    def __init__(self, seed=None):
        np.random.seed(seed)

    def action(self, state):
        steering = np.random.uniform(-1, 1)
        acc = np.random.uniform(0, 1)
        brake = np.random.uniform(0, 1)
        return [steering, acc, brake]

class eGreedyPolicy():

    def __init__(self, env, seed, eps):
        pass

    def action(self, state):
        pass
