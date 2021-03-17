from tqdm import tqdm
import config as cfg
from torch.utils.tensorboard import SummaryWriter
import os
import gym
import torch

from policy import NoExplorationPolicy, eGreedyPolicy
from dqn import DQN, DuelingDQN
from agent import Agent

from utils import get_latest_model, load_checkpoint
from gym_utils import getWrappedEnv
from collections import defaultdict


def evaluate_agents(dir):
    # load every agent in directory, play 1 episode in every seed, plot mean reward
    writer = SummaryWriter(log_dir=dir)
    files = sorted([filename for filename in os.listdir(
        dir) if filename.endswith(".pt")], key=lambda x: int(x[29:-10]))
    print(files)
    for filename in files:
        filename = os.path.join(dir, filename)
        checkpoint = torch.load(filename, map_location='cpu')
        frames = checkpoint["info"]["frames"]
        print(filename, frames)
    for filename in files:
        filename = os.path.join(dir, filename)
        checkpoint = torch.load(filename, map_location='cpu')
        env = getWrappedEnv(seed=checkpoint["info"]["seed"])
        dqn = DuelingDQN(env)

        load_checkpoint(dqn, filename, dqn.device)

        policy = NoExplorationPolicy(dqn)
        agent = Agent(dqn, policy, None)
        frames = checkpoint["info"]["frames"]
        rewards = eval(agent)
        mean_rewards = {str(s): sum(r) / len(r) for s, r in rewards.items()}
        writer.add_scalars("MeanReward", mean_rewards, frames)
    writer.close()


def eval(agent, EPISODES=10):
    rewards = defaultdict(list)
    env = getWrappedEnv()
    for seed in cfg.SEEDS:
        done = False
        env.seed(seed)
        state = env.reset()
        for i in tqdm(range(EPISODES), desc=str(seed)):
            while not done:
                action = agent.act(state)
                next_state, reward, done, meta = env.step(action)
                rewards[seed].append(reward)
                state = next_state
    return rewards
