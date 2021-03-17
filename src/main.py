import argparse
import os
import pickle

from train import train, assemble_training
from play import assemble_play, play
from evaluate import evaluate_agents
import config as cfg

from pyvirtualdisplay import Display

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "-p", help="Play mode. Displays one episode using given weights.", action='store_true')
parser.add_argument(
    "-d", help="Devbox mode. Uses pyvirtualdisplay.", action='store_true')
parser.add_argument(
    "-e", help="Evaluation mode. Evaluate all checkpoints in given --dir option.", action='store_true')

parser.add_argument("--weights", help="pytorch checkpoint file to load.")
parser.add_argument(
    "--dir", help="Location relative to models/ for saving checkpoints, buffer and tensorboard.")
parser.add_argument("--seed", help="Random seed.")


def main():
    args = parser.parse_args()

    if args.d:
        display = Display(visible=0, size=(1400, 900))
        display.start()

    if args.seed:
        seed = int(args.seed)
    else:
        seed = 0

    if args.p:
        env, agent = assemble_play(args.weights, seed)
        play(env, agent)

    elif args.e:
        p = os.path.join("models", args.dir)
        evaluate_agents(p)

    else:
        save_dir = os.path.join("models", args.dir)
        if not os.path.isdir(save_dir):
            print("Create directory", save_dir)
            os.mkdir(save_dir)

        print("Checkpoints and buffer will be saved into", save_dir)

        env, agent, episodes, frames = assemble_training(seed, args.weights)

        train(
            env,
            agent,
            seed,
            SAVE_DIR=save_dir,
            EPISODES=cfg.EPISODES,
            EPISODE_LENGTH=cfg.EPISODE_LENGTH,
            SKIP_FRAMES=cfg.SKIP_FRAMES,
            OFFSET_EP=episodes,
            OFFSET_FR=frames,
        )


if __name__ == "__main__":
    main()
