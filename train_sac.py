import imageio
import gymnasium as gym
import numpy as np
import gym_aloha
import torch
from stable_baselines3 import SAC
import cv2
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from gymnasium.wrappers import RecordEpisodeStatistics

env = gym.make(
    "gym_aloha/SO100TransferCube-v0",
    obs_type="so100_pixels_agent_pos",
    observation_width=64,
    observation_height=48,
)

env = RecordEpisodeStatistics(env)
# 1) vectorise
vec_env = DummyVecEnv([lambda: env])
# vec_env = DummyVecEnv([lambda: env], create_env=False)  # for SB3 ≥ 2.4
# 2) move channel first for ALL dict image keys
vec_env = VecTransposeImage(vec_env)


# 3) normalise only the joint vector, leave pixels raw
vec_env = VecNormalize(
    vec_env,
    norm_obs=True,
    norm_reward=False,
    clip_obs=10.0,
    # obs_keys=["agent_pos"],   # requires SB3 ≥ 2.4; otherwise custom wrapper
)

device = "mps" if torch.backends.mps.is_available() else "cpu"

log_dir = "logs/sac_so100"  # will hold TB files
new_logger = configure(log_dir, ["tensorboard", "stdout"])


model = SAC(
    policy="MultiInputPolicy",
    env=vec_env,
    learning_rate=1e-4,
    batch_size=128,  # drop to 64 if CPU is slow
    buffer_size=10_000,
    device=device,
    tensorboard_log=log_dir,
    # policy_kwargs    ={
    #     # keep default NatureCNN; change later if FPS is painful
    #     "features_extractor_kwargs": {"features_dim": 128},
    # },
)
model.set_logger(new_logger)
model.learn(50_000)
model.save("sac_so100_pixels_agentpos_50k")
vec_env.save("vec_normalize_stats_50k.pkl")
