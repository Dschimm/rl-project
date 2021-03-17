# Pretrained checkpoints

#### Hardware
Trained on:
 * Intel Xeon CPU E5-1650 v3 @ 3.50GHz (12 threads)
 * 1 nvidia RTX 2080 TI
 * 30 GB RAM

#### Time
Time per episode: ~3.5s

Thus, time for 150 episodes: ~6.07 days

#### Tuning hyperparameters

For models in `prio/`, we used a prioritized replay buffer,
otherwise we used a randomly sampling buffer.

All the hyperparameters used can be found in our [config](https://github.com/Dschimm/rl-project/blob/main/src/config.py).
Unfortunately, since training takes some time, we were not able to try a variety of hyperparameters.

#### Seeds
We used 4 different seeds (for env and numpy) for training and evaluation, namely
 * 42
 * 366
 * 533
 * 1337

## Training visualization

To click through our data yourself, you can run 
```
$ /home/dentarthur/rl-project> tensorboard --logdir models/tensorboard
```
#### Mean Reward and Loss per Frame

<img src="https://github.com/Dschimm/rl-project/blob/main/images/trainreward.svg" width="450" height="450" align="left">

<img src="https://github.com/Dschimm/rl-project/blob/main/images/trainloss.svg" width="450" height="450" align="left">
</br></br></br></br></br></br></br></br></br></br>
</br>

Note: Over time, epsilon decays. The car will therefore be faster and moves forward with a higher probability.

## Evaluation of checkpoints

TODO - eval and plots (and table)

## Conclusion

As the plots show, the agent has not learned a policy that solves the environment.
To be more precise, it has learned to drive straight forward and hopes that most of the area in front is road, thus maximizing its reward before hitting the void.

We think this is due to the small likelihood of ever successfully exploring how to drive a curve when using epsilon greedy exploration.

Normal agent (left) and agent with 0.5 eps (right):

<img src="https://github.com/Dschimm/rl-project/blob/main/images/1337.gif" width="200" height="200" align="left"> 

<img src="https://github.com/Dschimm/rl-project/blob/main/images/1337epsgreedy.gif" width="200" height="200" align="left"> 
</br>
</br>
</br>
</br>
</br></br></br></br></br>

With more time (and less training time) we would try to:
 1. implement a noisy DQN for a different exploration
 2. shape the reward function to punish driving on grass for too long
 3. increase downsampling for less information input
 4. Remove going straight from action space (only allowing gas + direction)
