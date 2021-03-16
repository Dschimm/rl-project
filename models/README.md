# Pretrained checkpoint

Trained on:
 * Intel Xeon CPU E5-1650 v3 @ 3.50GHz (12 threads)
 * 1 nvidia RTX 2080 TI
 * 30 GB RAM

Time per episode: ~3.5s

For models in `prio/`, we used Prioritized Replay Buffer
Otherwise we used a normal buffer.

All the hyperparameters used can be found in our [config](https://github.com/Dschimm/rl-project/src/config.py).
Unfortunately, since training takes some time, we were not able to try a variety of hyperparameters.

We used 4 different seeds (for env and numpy) for training and evaluation, namely
 * 42
 * 366
 * 533
 * 1337

## Training visualization


## Evaluation of checkpoints
