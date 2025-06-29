import imageio
import gymnasium as gym
import numpy as np
import gym_aloha
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from gymnasium.wrappers import RecordEpisodeStatistics 

# STEP 1: Create the EXACT same environment pipeline as training
env = gym.make(
    "gym_aloha/SO100TransferCube-v0",
    obs_type="so100_pixels_agent_pos", 
    observation_width=64,
    observation_height=48,
)

env = RecordEpisodeStatistics(env)
# STEP 2: Apply the EXACT same wrappers in the EXACT same order
vec_env = DummyVecEnv([lambda: env])
vec_env = VecTransposeImage(vec_env)
demo_env = VecNormalize(
    vec_env,
    norm_obs=True,
    norm_reward=False,
    clip_obs=10.0,
    training=False,  # ← IMPORTANT: Set to False for demo/inference
)

device = "mps" if torch.backends.mps.is_available() else "cpu"

# STEP 3: Load the model
model = SAC.load("sac_so100_pixels_agentpos_5k", env=demo_env, device=device)

# STEP 4: Load the normalization statistics
demo_env = VecNormalize.load("vec_normalize_stats_5k.pkl", demo_env)

print("Model and normalization stats loaded successfully!")

# STEP 5: Run the demo
observation = demo_env.reset()
frames = []
max_steps = 1000
steps = 0
action, _ = model.predict(observation, deterministic=True)
step_result = demo_env.step(action)
print(f"demo_env.step() returns {len(step_result)} values")

print("Starting demo...")

ret = 0
while steps < max_steps:
    # Get action from trained model
    action, _ = model.predict(observation, deterministic=True)
    
    # Step the environment
    step_result = demo_env.step(action)
    if len(step_result) == 4:
        observation, reward, done, info = step_result
        terminated = truncated = done
    else:
        observation, reward, terminated, truncated, info = step_result
    
    ret += reward[0]  # VecEnv returns arrays, so we take the first element
    # Get frame from the wrapped environment
    frame = demo_env.render()
    if frame is not None:
        frames.append(frame)
    
    steps += 1
    
    # Print progress
    if steps % 100 == 0:
        print(f"Step {steps}/{max_steps}")
    
    # Check if episode ended
    if terminated[0] or truncated[0]:  # VecEnv returns arrays
        print(f"Episode ended at step {steps}")
        print(f"Total reward: {ret}")
        ret = 0  # Reset return for the next episode
        observation = demo_env.reset()

demo_env.close()

# STEP 6: Save video
if frames:
    print(f"Saving {len(frames)} frames to video...")
    imageio.mimsave("so100_sac_demo_5k_n.mp4", np.stack(frames), fps=25)
    print("Saved video to so100_sac_demo_5k_n.mp4")
else:
    print("No frames captured!")