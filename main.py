import numpy as np
import pybullet_envs
import gym

ENV_ID = 'HalfCheetahBulletEnv-v0'
env = gym.make(ENV_ID)

print(env.observation_space)
print(env.action_space)
