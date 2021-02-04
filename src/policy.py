import numpy as np

class RandomPolicy():
    
    def __init__(self, seed=None):
        np.random.seed(seed)

    def action(self, state):
        return 0
        #return np.random.randint(9)

class eGreedyPolicy():

    def __init__(self, env, seed, eps):
        pass

    def action(self, state):
        pass
