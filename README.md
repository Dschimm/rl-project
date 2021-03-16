# 🏁🦆🏎️ RUBBER DUCK RACING 🏎️🦆🏁
<img src="https://github.com/Dschimm/rl-project/blob/main/images/racing_mascot.jpg" width="250" height="250" align="right">

[Reinforcement Learning Project for RL2020@LUH](https://github.com/automl-edu/RL_lecture)

We decided to build an agent that solves the CarRacing-v0 gym environment.

To improve learning, we pre-processed the states with:
 * Greyscaling
 * Framestacking
 * Resizing to 64\*64 pixels

Additionally, we implemented [macro-features](https://github.com/Dschimm/rl-project/blob/main/src/gym_utils.py#L15).

Our agent implements with the following features:
 * DQN
 * DoubleDQN
 * DuelingDQN
 * Prioritized Replay Buffer
 * Decaying  ε-greedy exploration

## Installation

### Docker

Build container 
```
docker build .
```
OR
```
docker-compose up
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

## Authors

[Jim Rhotert](https://github.com/Dschimm) & [Sebastian Döhler](https://github.com/sebidoe)
