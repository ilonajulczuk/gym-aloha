import imageio
import gymnasium as gym
import numpy as np
import gym_aloha
import torch
import os
import argparse
from stable_baselines3 import SAC
import cv2
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from gymnasium.wrappers import RecordEpisodeStatistics

def create_environment():
    """Create and configure the training environment."""
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
    
    return vec_env


def create_model(vec_env, log_dir):
    """Create and configure the SAC model."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    new_logger = configure(log_dir, ["tensorboard", "stdout"])

    model = SAC(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=3e-4,        # Keep your current rate
        buffer_size=50_000,        # Increase this (big stability gain)
        batch_size=256,            # Increase this (stability)
        ent_coef='auto',
        target_entropy=-0.5,       # Fix entropy (stop the chaos)
        device=device,
        tensorboard_log=log_dir,
    )
    model.set_logger(new_logger)
    return model


def load_checkpoint(checkpoint_path, vec_env_stats_path, vec_env, log_dir):
    """Load model and vectorized environment from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Extract step count from checkpoint filename
    import re
    match = re.search(r'(\d+)_steps\.zip$', checkpoint_path)
    start_steps = int(match.group(1)) if match else 0
    
    # Load the model
    model = SAC.load(checkpoint_path)
    
    # Reconfigure the environment and logger
    model.set_env(vec_env)
    new_logger = configure(log_dir, ["tensorboard", "stdout"])
    model.set_logger(new_logger)
    
    # Load vectorized environment stats if available
    if vec_env_stats_path and os.path.exists(vec_env_stats_path):
        print(f"Loading VecNormalize stats from: {vec_env_stats_path}")
        vec_env = VecNormalize.load(vec_env_stats_path, vec_env)
        model.set_env(vec_env)
    else:
        print("VecNormalize stats file not found, using fresh normalization")
    
    print(f"Checkpoint loaded from step {start_steps}")
    return model, vec_env, start_steps


def create_callbacks(vec_env, save_freq=2000):
    """Create training callbacks for model and environment checkpointing."""
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="./checkpoints/",
        name_prefix="sac_so100_get_cube",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    return checkpoint_callback

class StageBasedTraining:
    def __init__(self, model, vec_env, callback=None, start_steps=0):
        self.model = model
        self.vec_env = vec_env
        self.callback = callback
        self.start_steps = start_steps
        
        # Define stage boundaries
        self.stage1_end = 20000
        self.stage2_end = 35000
        self.stage3_end = 55000

    def train(self):
        current_steps = self.start_steps
        
        # Stage 1: High exploration phase (0-20k steps)
        if current_steps < self.stage1_end:
            remaining_stage1 = self.stage1_end - current_steps
            print(f"Stage 1: Exploration phase (continuing from step {current_steps}, {remaining_stage1} steps remaining)")
            self.model.target_entropy = -0.5  # High exploration
            self.model.learning_rate = 3e-4   # Fast learning
            self.model.learn(remaining_stage1, callback=self.callback)
            current_steps = self.stage1_end
        else:
            print(f"Stage 1: Already completed (started from step {current_steps})")
        
        # Stage 2: Balanced phase (20k-35k steps)  
        if current_steps < self.stage2_end:
            remaining_stage2 = self.stage2_end - current_steps
            print(f"Stage 2: Balanced phase (continuing from step {current_steps}, {remaining_stage2} steps remaining)")
            self.model.target_entropy = -3.0  # Low exploration
            self.model.learning_rate = 1e-4   # Medium learning
            self.model.learn(remaining_stage2, callback=self.callback)
            current_steps = self.stage2_end
        else:
            print(f"Stage 2: Already completed (started from step {current_steps})")
        
        # Stage 3: Exploitation phase (35k-50k steps)
        if current_steps < self.stage3_end:
            remaining_stage3 = self.stage3_end - current_steps
            print(f"Stage 3: Exploitation phase (continuing from step {current_steps}, {remaining_stage3} steps remaining)")
            self.model.target_entropy = -3.0  # Low exploration
            self.model.learning_rate = 5e-5   # Slow learning
            self.model.learn(remaining_stage3, callback=self.callback)
        else:
            print(f"Stage 3: Already completed (started from step {current_steps})")
            print("All training stages completed!")

def train_model(checkpoint_path=None, vec_env_stats_path=None, total_steps=50000, save_freq=1000):
    """Main training function with optional checkpoint loading."""
    log_dir = "logs/sac_so100"  # will hold TB files
    
    # Create environment
    vec_env = create_environment()
    
    # Create or load model
    start_steps = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        model, vec_env, start_steps = load_checkpoint(checkpoint_path, vec_env_stats_path, vec_env, log_dir)
        print(f"Resuming training from checkpoint: {checkpoint_path}")
    else:
        model = create_model(vec_env, log_dir)
        print("Starting training from scratch")
    
    # Create callbacks
    combined_callback = create_callbacks(vec_env, save_freq)

    # Stage-based training with checkpoint awareness
    trainer = StageBasedTraining(model, vec_env, callback=combined_callback, start_steps=start_steps)
    trainer.train()

    # Save final model and environment stats
    model.save("sac_so100_pixels_agentpos")
    vec_env.save("vec_normalize_stats_pixels_agentpos.pkl")
    print("Training completed and model saved!")


def list_available_checkpoints(checkpoint_dir="./checkpoints/"):
    """List available model checkpoints and their corresponding VecNormalize stats."""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist")
        return []
    
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith("sac_so100_pixels_agentpos") and file.endswith(".zip"):
            checkpoint_path = os.path.join(checkpoint_dir, file)
            
            # Look for corresponding VecNormalize stats
            import re
            match = re.search(r'(\d+)_steps\.zip$', file)
            if match:
                steps = match.group(1)
                vec_stats_file = f"vec_normalize_pixels_agentpos_stats_{steps}.pkl"
                vec_stats_path = os.path.join(checkpoint_dir, vec_stats_file)
                vec_stats_exists = os.path.exists(vec_stats_path)
            else:
                vec_stats_path = None
                vec_stats_exists = False
            
            checkpoints.append({
                "checkpoint": checkpoint_path,
                "vec_stats": vec_stats_path if vec_stats_exists else None,
                "steps": steps if match else "unknown"
            })
    
    if checkpoints:
        print("Available checkpoints:")
        for cp in sorted(checkpoints, key=lambda x: int(x["steps"]) if x["steps"].isdigit() else 0):
            print(f"  Steps: {cp['steps']}")
            print(f"    Model: {cp['checkpoint']}")
            print(f"    VecNormalize: {cp['vec_stats'] if cp['vec_stats'] else 'Not found'}")
            print()
    else:
        print("No checkpoints found")
    
    return checkpoints


def main():
    """Main function with argument parsing for checkpoint loading."""
    parser = argparse.ArgumentParser(description="Train SAC model with optional checkpoint loading")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None,
        help="Path to model checkpoint to resume from (e.g., './checkpoints/sac_so100_pixels_agentpos_20000_steps.zip')"
    )
    parser.add_argument(
        "--vec-env-stats", 
        type=str, 
        default=None,
        help="Path to VecNormalize stats file (e.g., './checkpoints/vec_normalize_pixels_agentpos_stats_20000.pkl')"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=50000,
        help="Total training steps (default: 50000)"
    )
    parser.add_argument(
        "--save-freq", 
        type=int, 
        default=1000,
        help="Frequency of saving checkpoints (default: 1000)"
    )
    parser.add_argument(
        "--list-checkpoints", 
        action="store_true",
        help="List available checkpoints and exit"
    )
    
    args = parser.parse_args()
    
    # List checkpoints if requested
    if args.list_checkpoints:
        list_available_checkpoints()
        return
    
    # Auto-detect vec_env_stats path if not provided but checkpoint is given
    if args.checkpoint and not args.vec_env_stats:
        # Extract step number from checkpoint filename
        import re
        match = re.search(r'(\d+)_steps\.zip$', args.checkpoint)
        if match:
            steps = match.group(1)
            args.vec_env_stats = f"./checkpoints/vec_normalize_pixels_agentpos_stats_{steps}.pkl"
            print(f"Auto-detected VecNormalize stats path: {args.vec_env_stats}")
    
    train_model(args.checkpoint, args.vec_env_stats, args.steps, args.save_freq)


if __name__ == "__main__":
    main()

# Usage
# trainer = StageBasedTraining(model, env, callback=combined_callback)
# trainer.train()

# model.save("sac_so100_pixels_agentpos")
# vec_env.save("vec_normalize_stats_pixels_agentpos.pkl")

# 1. Train from scratch:
#    python train_sac.py
#
# 2. Resume from a specific checkpoint:
#    python train_sac.py --checkpoint ./checkpoints/sac_so100_pixels_agentpos_20000_steps.zip
#
# 3. Resume with custom VecNormalize stats:
#    python train_sac.py --checkpoint ./checkpoints/sac_so100_pixels_agentpos_20000_steps.zip --vec-env-stats ./checkpoints/vec_normalize_pixels_agentpos_stats_20000.pkl
#
# 4. List available checkpoints:
#    python -c "from train_sac import list_available_checkpoints; list_available_checkpoints()"
