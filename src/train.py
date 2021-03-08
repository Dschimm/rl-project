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
from policy import RandomPolicy, eGreedyPolicy, eGreedyPolicyDecay
from dqn import DQN, DuelingDQN
from agent import Agent, DDQNAgent

from utils import get_latest_model, save_checkpoint, load_checkpoint, get_cuda_device
from gym_utils import getWrappedEnv

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

seeds = [
    42,
    366,
    533,
    1337,
]


def assemble_training(seed, weights=None, lr=0.01, er=1):
    if weights:
        checkpoint = torch.load(weights)
        env = getWrappedEnv(seed=checkpoint["info"]["seed"])
        dqn = DuelingDQN(env, lr=lr)
        eval_net = DuelingDQN(env)

        load_checkpoint(dqn, weights, dqn.device)
        load_checkpoint(eval_net, weights, dqn.device)

        policy = eGreedyPolicyDecay(env, seed, checkpoint["info"]["er"], er, 0.1, 25e4, dqn)
        buffer = PrioritizedReplayBuffer()
        agent = DDQNAgent(dqn, eval_net, policy, buffer)
        with open(checkpoint["info"]["buffer"], "rb") as f:
            preloaded_buffer = pickle.load(f)
        agent.buffer = preloaded_buffer
        print(
            "Resume training at Episode",
            checkpoint["info"]["episodes"],
            "after",
            checkpoint["info"]["frames"],
            "frames.",
        )
        return env, agent, checkpoint["info"]["episodes"], checkpoint["info"]["frames"]

    env = getWrappedEnv(seed=seed)
    dqn = DuelingDQN(env, lr=lr)
    eval_net = DuelingDQN(env)

    policy = eGreedyPolicyDecay(env, seed, er, er, 0.1, 25e4, dqn)
    buffer = PrioritizedReplayBuffer()
    agent = DDQNAgent(dqn, eval_net, policy, buffer)
    return env, agent, 0, 0


def train(
    env,
    agent,
    seed,
    SAVE_DIR="models/",
    EPISODES=10000,
    EPISODE_LENGTH=10000,
    SKIP_FRAMES=80000,
    OFFSET_EP=0,
    OFFSET_FR=0,
):

    writer = SummaryWriter(log_dir=SAVE_DIR, comment=str(seed))
    writer.add_graph(
        agent.actor_model,
        torch.tensor(env.reset()).unsqueeze(0).float().to(agent.actor_model.device),
    )

    losses = []
    rewards = []
    frames = OFFSET_FR
    for i in tqdm(range(OFFSET_EP, EPISODES + OFFSET_EP), desc="Episodes"):
        state = env.reset()
        for j in tqdm(range(EPISODE_LENGTH), desc="Episode length"):
            action = agent.act(state)
            next_state, reward, done, meta = env.step(action)
            frames += 4  # frame stacking

            rewards.append(reward)
            state = next_state

            agent.fill_buffer((state, action, reward, done, next_state))
            if frames > SKIP_FRAMES:
                loss = agent.update()
                agent.sync_nets()
                agent.policy.decay_eps()
                losses.append(loss)

            if done:
                break

        if i % 1 == 0:
            writer.add_scalar("MeanReward", np.mean(rewards), i)
            writer.add_scalar("Frames", frames, i)
            writer.add_scalar("Loss", np.mean(losses), i)
            writer.flush()
            losses.clear()
            rewards.clear()

        if (i + 1) % 10 == 0:
            buffer_dir = os.path.join(SAVE_DIR, "buffer_seed_" + str(seed) + ".pkl")
            save_checkpoint(
                agent.actor_model,
                "Duelingddqn_seed_" + str(seed) + "_EPISODE_" + str(i + 1),
                loc=SAVE_DIR,
                info={
                    "seed": seed,
                    "lr": agent.actor_model.lr,
                    "er": agent.policy.eps,
                    "episodes": i,
                    "frames": frames,
                    "buffer": buffer_dir,
                },
            )
            with open(buffer_dir, "wb") as f:
                pickle.dump(agent.buffer, f)
    writer.close()
    return
