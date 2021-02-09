import gym
from gym.spaces.discrete import Discrete

# [steering, acc, brake]
# [(-1,1), (0,1), (0,1)]


class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.actions = {
            0: [0, 0.5, 0],  # Gas50
            1: [-0.5, 0.5, 0],  # GasLeft
            2: [0.5, 0.5, 0],  # GasRight
            3: [0, 0, 0.7],  # Brake
            4: [-0.5, 0, 0.5],  # BrakeLeft
            5: [0.5, 0, 0.5],  # BrakeRight
            6: [-0.7, 0, 0],  # TurnLeft
            7: [0.7, 0, 0],  # TurnRight
            8: [0, 0, 0],  # Nothing
        }
        self.action_space = Discrete(len(self.actions))

    def action(self, i):
        return self.actions[i]
