import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

import pickle

import gym

from buffer import ReplayBuffer, PrioritizedReplayBuffer
from policy import RandomPolicy, eGreedyPolicy
from dqn import DQN, DuelingDQN
from agent import Agent, DDQNAgent

from utils import get_latest_model, save_checkpoint, load_checkpoint, get_cuda_device
from gym_utils import getWrappedEnv

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
# from pyvirtualdisplay import Display

# display = Display(visible=0, size=(1400, 900))
# display.start()

seeds = [
    42,
    366,
    533,
    1337,
]

learning_rates = [0.01, 0.001]


def train(env, agent, EPISODES=10000, EPISODE_LENGTH=10000, SKIP_FRAMES=80000, BATCH_SIZE=64):

    rewards = []
    rewards100 = []
    frames = 0
    for i in range(EPISODES):
        state = env.reset()
        print("Episode", i)
        for j in range(EPISODE_LENGTH):
            # env.render()
            print(".", end="", flush=True)
            action = agent.act(state)
            next_state, reward, done, meta = env.step(action)
            frames += 4 # frameskipping

            rewards.append(reward)
            rewards100.append(reward)
            state = next_state

            agent.fill_buffer((state, action, reward, done, next_state))
            """
            if frames % 10000 == 0:
                with open("models/buffer" + str(frames + 40000) + ".pkl", "wb+") as f:
                    pickle.dump(agent.buffer, f)
            """
            if frames > SKIP_FRAMES and len(agent.buffer) >= BATCH_SIZE:
                agent.update()

            if done:
                break

        if i % 1 == 0:
            writer.add_scalar("MeanReward", np.mean(rewards), i)

            print("Frames:", frames)
            print("Mean reward:", np.mean(rewards))
            save_checkpoint(
                agent.actor_model,
                "Duelingddqn",
                frames=frames,
                mean_reward=np.mean(rewards),
                overwrite=True,
            )
            rewards.clear()

        if i % 2 == 0:
            print("Frames:", frames)
            print("Mean reward:", np.mean(rewards100))
            save_checkpoint(
                agent.actor_model,
                "Duelingddqn_" + str(i),
                frames=frames,
                mean_reward=np.mean(rewards100),
            )
            rewards100.clear()

    save_checkpoint(agent.model, "dqn")
    print(frames)
    return


if __name__ == "__main__":
    with open("models/buffer80000.pkl", "rb") as f:
        preloaded_buffer = pickle.load(f)

    for seed in seeds:
        env = getWrappedEnv(seed=seed)

        dqn = DuelingDQN(env)
        eval_net = DuelingDQN(env)
        policy = eGreedyPolicy(env, seed, 0.1, dqn)
        buffer = PrioritizedReplayBuffer(seed)
        agent = DDQNAgent(dqn, eval_net, policy, buffer)
        agent.buffer = preloaded_buffer
        train(env, agent, SKIP_FRAMES=10)
