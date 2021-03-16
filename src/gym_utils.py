import gym
from gym.spaces.discrete import Discrete

from gym.wrappers.resize_observation import ResizeObservation
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.gray_scale_observation import GrayScaleObservation

import config as cfg
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


def getWrappedEnv(name="CarRacing-v0", seed=42):
    env = gym.make(name, verbose=0)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, cfg.RESIZE_SHAPE)
    env = ActionWrapper(env)
    env = FrameStack(env, cfg.FRAMESTACK)
    env.seed(seed)
    return env