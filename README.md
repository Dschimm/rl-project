# 🏁🦆🏎️ RUBBER DUCK RACING 🏎️🦆🏁
<img src="https://github.com/Dschimm/rl-project/blob/main/images/racing_mascot.jpg" width="250" height="250" align="right">

[Reinforcement Learning Project for RL2020@LUH](https://github.com/automl-edu/RL_lecture)

We decided to build an agent that solves the CarRacing-v0 gym environment.

To improve learning, we pre-processed the states with:
 * Greyscaling
 * Framestacking
 * Resizing to 64\*64 pixels

Additionally, we implemented [macro-actions](https://github.com/Dschimm/rl-project/blob/main/src/gym_utils.py#L15).

Our agent implements the following features:
 * DQN / DuelingDQN
 * DoubleDQN
 * Prioritized Replay Buffer
 * Decaying  ε-greedy exploration

For pretrained checkpoints, their training and evaluation, click [here](https://github.com/Dschimm/rl-project/blob/main/models/).

## Installation

### Docker

Build container 
```
docker build . -t rl:latest
```

### Pip and venv

Create virtual environment
```
python -m venv .venv
```

Activate it
```
source .venv/bin/activate
```

Install required packages
```
pip install -r requirements.txt
```
## Running the code

Run the code from parent directory via main.py:
```
$ /home/dentarthur/rl-project> python src/main.py --help

usage: main.py [-h] [-p] [-d] [--weights WEIGHTS] [--dir DIR] [--seed SEED]

optional arguments:
  -h, --help         show this help message and exit
  -p                 Play mode. Displays one episode using given weights.
  -d                 Devbox mode. Uses pyvirtualdisplay.
  -e                 Evaluation mode. Evaluate all checkpoints in given --dir option.
  --weights WEIGHTS  pytorch checkpoint file to load.
  --dir DIR          Location relative to models/ for saving checkpoints, buffer and tensorboard.
  --seed SEED        Random seed.
```
#### Examples:

Training from scratch on headless server; seed 42, saving in models/exp42:

```
$ /home/dentarthur/rl-project> python src/main.py -d --seed 42 --dir exp42
```

Resume above training:

```
$ /home/dentarthur/rl-project> python src/main.py -d --dir exp42 --weights models/exp42/latest.pt
```

Let a pretrained agent play:


```
$ /home/dentarthur/rl-project> python src/main.py -p --weights models/exp42/latest.pt
```

Evaluate all checkpoints from exp42:

```
$ /home/dentarthur/rl-project> python src/main.py -e --dir exp42
```

Note: This will create tensorboard plots and not print anything!

## Authors

[Jim Rhotert](https://github.com/Dschimm) & [Sebastian Döhler](https://github.com/sebidoe)
