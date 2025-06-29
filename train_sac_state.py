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

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from gymnasium.wrappers import RecordEpisodeStatistics

env = gym.make(
    "gym_aloha/SO100TransferCube-v0",
    obs_type="so100_state",
    observation_width=64,
    observation_height=48,
)

env = RecordEpisodeStatistics(env)
# 1) vectorise
vec_env = DummyVecEnv([lambda: env])
# vec_env = DummyVecEnv([lambda: env], create_env=False)  # for SB3 ≥ 2.4
# 2) move channel first for ALL dict image keys
# vec_env = VecTransposeImage(vec_env)


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
    policy="MlpPolicy",
    env=vec_env,
    learning_rate=1e-4,        # Keep your current rate
    buffer_size=50_000,        # Increase this (big stability gain)
    batch_size=256,            # Increase this (stability)
    ent_coef='auto',
    target_entropy=-2.0,       # Fix entropy (stop the chaos)
    device=device,
    tensorboard_log=log_dir,
    # policy_kwargs    ={
    #     # keep default NatureCNN; change later if FPS is painful
    #     "features_extractor_kwargs": {"features_dim": 128},
    # },
)
model.set_logger(new_logger)


model_checkpoint = CheckpointCallback(
    save_freq=10000,
    save_path='./checkpoints/',
    name_prefix='sac_so100'
)

# Vec env checkpoint
class VecEnvSaveCallback(BaseCallback):
    def __init__(self, save_freq, save_path, vec_env):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.vec_env = vec_env

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.vec_env.save(f"{self.save_path}/vec_normalize_stats_{self.n_calls}.pkl")
        return True

vec_env_checkpoint = VecEnvSaveCallback(10000, './checkpoints/', vec_env)

# Combine both callbacks
combined_callback = CallbackList([model_checkpoint, vec_env_checkpoint])

model.learn(50_000, callback=combined_callback)


# model.learn(50_000)
model.save("sac_so100_pixels_agentpos_50k")
vec_env.save("vec_normalize_stats_50k.pkl")
