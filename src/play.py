import os
import gym
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym_utils import ActionWrapper

from buffer import ReplayBuffer, PrioritizedReplayBuffer
from policy import NoExplorationPolicy
from dqn import DQN
from agent import Agent

from utils import get_latest_model, save_checkpoint, load_checkpoint, get_cuda_device


def play(env, agent):
    done = False
    state = env.reset()
    while not done:
        env.render()

        action = agent.act(state)
        next_state, reward, done, meta = env.step(action)

        state = next_state
    return


if __name__ == "__main__":
    seed = 42
    env = ActionWrapper(GrayScaleObservation(gym.make('CarRacing-v0')))
    env.seed(seed)
    dqn = DQN(env)
    load_checkpoint(dqn, get_latest_model(prefix="dqn"), dqn.device)
    policy = NoExplorationPolicy(dqn)
    agent = Agent(dqn, policy, None)
    play(env, agent)
