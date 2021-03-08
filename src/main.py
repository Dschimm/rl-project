import argparse
import os
import pickle

from train import train, assemble_training
from play import assemble_play, play

# from pyvirtualdisplay import Display

# display = Display(visible=0, size=(1400, 900))
# display.start()

parser = argparse.ArgumentParser(description="")
parser.add_argument("-t", help="Training mode.", action='store_true')
parser.add_argument("-p", help="Play mode.", action='store_true')

parser.add_argument("--weights", help="Checkpoint from which to resume training.")
parser.add_argument("--dir", help="Checkpoint directory.")
parser.add_argument("--seed", help="Random seed.")
parser.add_argument("--buffer", help="Buffer checkpoint.")


def main():
    args = parser.parse_args()
    if args.seed:
            seed = int(args.seed)
    else:
        seed = 0
    if args.p:
        env, agent = assemble_play(args.weights, seed)
        play(env, agent)
    else:


        save_dir = os.path.join("models", args.dir)
        if not os.path.isdir(save_dir):
            print("Create directory", save_dir)
            os.mkdir(save_dir)
        print("Checkpoints and buffer will be saved into", save_dir)

        if not args.buffer:
            buffer_dir = "models/buffer80000.pkl"
        else:
            buffer_dir = os.path.join("models", args.buffer)
        if os.path.exists(buffer_dir):
            print("Loading", buffer_dir)
            with open(buffer_dir, "rb") as f:
                preloaded_buffer = pickle.load(f)

        env, agent, episodes, frames = assemble_training(seed, preloaded_buffer, args.weights)

        train(
            env,
            agent,
            seed,
            SAVE_DIR=save_dir,
            EPISODES=100,
            EPISODE_LENGTH=3,
            SKIP_FRAMES=10,
            OFFSET_EP=episodes,
            OFFSET_FR=frames,
        )


if __name__ == "__main__":
    main()