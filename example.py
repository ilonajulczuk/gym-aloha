import imageio
import gymnasium as gym
import numpy as np
import gym_aloha

from stable_baselines3 import SAC

from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize



env = gym.make("gym_aloha/SO100TransferCube-v0", obs_type="so100_pixels_agent_pos", observation_width=64, observation_height=48)

observation, info = env.reset()
frames = []


for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    frames.append(image)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
imageio.mimsave("so100_example.mp4", np.stack(frames), fps=25)
