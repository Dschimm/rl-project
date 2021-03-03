import os
import gym

from policy import NoExplorationPolicy
from dqn import DQN, DuelingDQN
from agent import Agent

from utils import get_latest_model, load_checkpoint
from gym_utils import getWrappedEnv


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
    env = getWrappedEnv(seed=seed)
    dqn = DuelingDQN(env)
    load_checkpoint(dqn, get_latest_model(prefix="Duelingddqn"), dqn.device)
    policy = NoExplorationPolicy(dqn)
    agent = Agent(dqn, policy, None)
    play(env, agent)
