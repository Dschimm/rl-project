import os
import gym

from policy import NoExplorationPolicy
from dqn import DQN, DuelingDQN
from agent import Agent

from utils import get_latest_model, load_checkpoint
from gym_utils import getWrappedEnv


def assemble_play(weights, seed):
    """
    Prepare the environment for playing an episode with a fixed policy to 
    evaluate the learned model.
    """
    env = getWrappedEnv(seed=seed)
    dqn = DuelingDQN(env)
    load_checkpoint(dqn, weights, dqn.device)
    policy = NoExplorationPolicy(dqn)
    agent = Agent(dqn, policy, None)
    return env, agent


def play(env, agent):
    """
    Play 10 episodes with the given agent and evironment.
    """
    for i in range(10):
        done = False
        state = env.reset()
        rewards = []
        while not done:
            env.render()

            action = agent.act(state)
            next_state, reward, done, meta = env.step(action)
            rewards.append(reward)
            state = next_state

        print("Rewards:", sum(rewards))
    return
