import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

import pickle
import argparse

import gym

from buffer import ReplayBuffer, PrioritizedReplayBuffer
from policy import RandomPolicy, eGreedyPolicy
from dqn import DQN, DuelingDQN
from agent import Agent, DDQNAgent

from utils import get_latest_model, save_checkpoint, load_checkpoint, get_cuda_device
from gym_utils import getWrappedEnv

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from pyvirtualdisplay import Display

display = Display(visible=0, size=(1400, 900))
display.start()


parser = argparse.ArgumentParser(description='')
parser.add_argument('--weights', help='Checkpoint from which to resume training.')
parser.add_argument('--dir', help='Checkpoint directory.')
parser.add_argument('--seed', help='Random seed.')
parser.add_argument('--buffer', help='Buffer checkpoint.')



seeds = [
    42,
    366,
    533,
    1337,
]

learning_rates = [0.01, 0.001]

def assemble_training(seed, buffer, weights=None, lr=0.01, er=0.1):
    env = getWrappedEnv(seed=seed)
    dqn = DuelingDQN(env, lr=lr)
    eval_net = DuelingDQN(env)
    if weights:
        load_checkpoint(dqn, weights, dqn.device)
        load_checkpoint(eval_net, weights, dqn.device)
    policy = eGreedyPolicy(env, seed, er, dqn)
    buffer = PrioritizedReplayBuffer(seed)
    agent = DDQNAgent(dqn, eval_net, policy, buffer)
    agent.buffer = buffer
    return env, agent


def train(env, agent, seed, SAVE_DIR="models/", EPISODES=10000, EPISODE_LENGTH=10000, SKIP_FRAMES=80000, BATCH_SIZE=64):

    rewards = []
    rewards10 = []
    frames = 0
    for i in tqdm(range(EPISODES), desc="Episodes"):
        state = env.reset()
        for j in tqdm(range(EPISODE_LENGTH), desc="Episode length"):
            action = agent.act(state)
            next_state, reward, done, meta = env.step(action)
            frames += 4 # frameskipping

            rewards.append(reward)
            rewards10.append(reward)
            state = next_state

            agent.fill_buffer((state, action, reward, done, next_state))

            if frames > SKIP_FRAMES and len(agent.buffer) >= BATCH_SIZE:
                agent.update()

            if done:
                break

        if i % 1 == 0:
            writer.add_scalar("MeanReward", np.mean(rewards), i)
            writer.add_scalar("Frames", frames)
            writer.flush()

        if i % 9 == 0:
            save_checkpoint(
                agent.actor_model,
                "Duelingddqn_seed_" + str(seed) + "_EPISODE_" + str(i + 1),
                frames=frames,
                mean_reward=np.mean(rewards10),
                loc=SAVE_DIR
            )
            rewards10.clear()
            with open(os.path.join(SAVE_DIR, "buffer_seed_" + str(seed) + ".pkl"), "wb") as f:
                pickle.dump(agent.buffer, f)

    return


if __name__ == "__main__":
    args = parser.parse_args()
    weights = args.weights
    seed = int(args.seed)

    save_dir = os.path.join("models", args.dir)
    if not os.path.isdir(save_dir):
        print("Create directory", save_dir)
        os.mkdir(save_dir)
    print("Checkpoints and buffer will be saved into", save_dir)

    if not args.buffer:
        buffer_dir = "models/buffer80000.pkl"
    buffer_dir = os.path.join("models", args.buffer)
    if os.path.exists(buffer_dir):
        print("Loading", buffer_dir)
        with open(buffer_dir, "rb") as f:
            preloaded_buffer = pickle.load(f)

    env, agent = assemble_training(seed, preloaded_buffer, weights)

    writer = SummaryWriter(log_dir=save_dir,comment=str(seed))
    writer.add_graph(agent.actor_model, torch.tensor(env.reset()).unsqueeze(0).float().to(dqn.device))


    train(env, agent, seed, SAVE_DIR=save_dir, EPISODES=100, SKIP_FRAMES=10)
    writer.close()