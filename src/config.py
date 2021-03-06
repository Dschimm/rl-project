# Hyperparameters
LEARNING_RATE = 0.01
DEFAULT_SEED = 42
DISCOUNT_FACTOR = 0.9

EPS_START = 1
EPS_END = 0.1
DECAY_STEPS = 25e4

BATCH_SIZE = 64

# Training
EPISODES = 100
EPISODE_LENGTH = 1000
SKIP_FRAMES = 80
TAU = 0.01

BUFFER_SIZE = 1e5

# Pre-processing
FRAMESTACK = 4
RESIZE_SHAPE = 64

SEEDS = [42, 366, 533, 1337]
