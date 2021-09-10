import torch
from algo import Trainer, play_mp4
import pybullet_envs
import pybullet
import gym
from SAC import SAC


ENV_ID = 'BipedalWalker-v3'
# ENV_ID = 'Pendulum-v0'
SEED = 0
REWARD_SCALE = 1.0
NUM_STEPS = 5 * 10 ** 5
# NUM_STEPS = 2 * 10 ** 3
EVAL_INTERVAL = 10 ** 3

env = gym.make(ENV_ID)
env_test = gym.make(ENV_ID)
print("state {}".format(*env.observation_space.shape))
print("act {}".format(*env.action_space.shape))

state_shape = 3
act_shape = 1

algo = SAC(
    state_shape=env.observation_space.shape,
    action_shape=env.action_space.shape,
    seed=SEED,
    reward_scale=REWARD_SCALE,
    start_steps=5 * 10**2,
)

trainer = Trainer(
    env=env,
    env_test=env_test,
    algo=algo,
    seed=SEED,
    num_steps=NUM_STEPS,
    eval_interval=EVAL_INTERVAL,
)

trainer.train()
# trainer.plot()
trainer.visualize()
# trainer.visualize()
# play_mp4()
