# build_and_train.py
"""
Train PPO agent in physics-based hiking environment with FIXED REWARD STRUCTURE.
Uses progress-based rewards instead of proximity-based rewards to prevent
the agent from getting stuck in resting behavior.
"""

import os
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from physics_hiking_env import RealisticHikingEnv
import gpxpy
import gpxpy.gpx
from datetime import datetime
import time
from tqdm import tqdm
import csv
import logging


OUTPUT_DIR = Path("outputs")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
TRAJ_DIR = OUTPUT_DIR / "trajectories"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
TRAJ_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------
# Callback to export GPX on success
# -------------------------------
class GPXExportCallback(BaseCallback):
    """
    Enhanced callback that:
    1. Exports GPX files for successful episodes 
    2. Tracks success rate properly for curriculum progression
    3. Provides real-time progress updates with success tracking
    4. Logs detailed training progress to files
    """
    def __init__(self, save_freq=1, verbose=0):
        super().__init__(verbose)
        self.episode = 0
        self.successful_episodes = 0
        self.total_episodes = 0
        self.progress_bar = None
        self.last_update_time = time.time()
        
        # Setup logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Training progress logger
        self.progress_logger = logging.getLogger('training_progress')
        self.progress_logger.setLevel(logging.INFO)
        if not self.progress_logger.handlers:
            handler = logging.FileHandler(log_dir / f"training_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.progress_logger.addHandler(handler)
        
        # CSV logger for detailed episode data
        self.csv_file = log_dir / f"episode_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'success', 'reward', 'length', 'curriculum_distance', 'curriculum_success_rate', 'timestamp'])
        
    def _on_training_start(self) -> None:
        """Initialize progress bar and log training start"""
        total_timesteps = self.locals.get('total_timesteps', 1000000)
        self.progress_bar = tqdm(
            total=total_timesteps,
            desc="🏔️ Training Hiking Agent",
            unit="steps", 
            ncols=120,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Success: {postfix}"
        )
        
        self.progress_logger.info(f"Training started with {total_timesteps} timesteps")
        return True
        
    def _on_step(self) -> bool:
        """Update progress bar and handle successful episodes"""
        # Update progress bar every 1000 steps or every 5 seconds
        current_time = time.time()
        if self.num_timesteps % 1000 == 0 or (current_time - self.last_update_time) > 5:
            if self.progress_bar:
                # Calculate how much to update progress bar
                steps_to_update = min(1000, self.progress_bar.total - self.progress_bar.n)
                if steps_to_update > 0:
                    self.progress_bar.update(steps_to_update)
                
                # Enhanced success tracking with curriculum info
                try:
                    env = self.training_env
                    if env and hasattr(env, 'envs') and hasattr(env.envs[0], 'curriculum_successes'):
                        # Curriculum learning info
                        successes = env.envs[0].curriculum_successes
                        attempts = max(1, env.envs[0].curriculum_attempts)
                        current_distance = env.envs[0].start_distance_meters
                        success_rate = successes / attempts * 100
                        
                        self.progress_bar.set_postfix_str(
                            f"Success: {success_rate:.1f}% @ {current_distance:.0f}m | "
                            f"Total: {self.successful_episodes}/{self.total_episodes} episodes"
                        )
                    else:
                        # Fallback to simple tracking
                        success_rate = (self.successful_episodes / max(1, self.total_episodes)) * 100
                        self.progress_bar.set_postfix_str(f"Success: {success_rate:.1f}% ({self.successful_episodes}/{self.total_episodes})")
                except Exception as e:
                    success_rate = (self.successful_episodes / max(1, self.total_episodes)) * 100
                    self.progress_bar.set_postfix_str(f"Success: {success_rate:.1f}% ({self.successful_episodes}/{self.total_episodes})")
                
            self.last_update_time = current_time
        
        # Check if episode ended and goal was reached
        if self.locals.get("dones") is not None and np.any(self.locals["dones"]):
            infos = self.locals["infos"]
            print(f"\n🔍 Episode end detected, checking {len(infos)} info dicts...")
            for i, info in enumerate(infos):
                print(f"🔍 Info {i}: keys = {list(info.keys())}")
                # Count every episode completion
                if "episode" in info:
                    self.total_episodes += 1
                    episode_reward = info.get("episode", {}).get("r", "unknown")
                    episode_length = info.get("episode", {}).get("l", "unknown")
                    
                    # Check if goal was reached (FIXED SUCCESS COUNTING)
                    goal_reached = info.get("reached_goal", False)
                    
                    if goal_reached:
                        # COUNT SUCCESS IMMEDIATELY when goal is reached
                        self.successful_episodes += 1
                        print(f"\n🎯 SUCCESS! Episode {self.total_episodes}: reward={episode_reward}, length={episode_length} steps")
                        
                        # Log success to file
                        self.progress_logger.info(f"SUCCESS - Episode {self.total_episodes}: reward={episode_reward}, length={episode_length}")
                    else:
                        print(f"\n❌ Episode {self.total_episodes} ended: reward={episode_reward}, length={episode_length} steps")
                        
                        # Log failure to file
                        self.progress_logger.info(f"FAILURE - Episode {self.total_episodes}: reward={episode_reward}, length={episode_length}")
                    
                    # Get curriculum info for logging
                    try:
                        env = self.training_env
                        if env and hasattr(env, 'envs') and hasattr(env.envs[0], 'start_distance_meters'):
                            curriculum_distance = env.envs[0].start_distance_meters
                            curriculum_success_rate = (env.envs[0].curriculum_successes / max(1, env.envs[0].curriculum_attempts)) * 100
                        else:
                            curriculum_distance = "unknown"
                            curriculum_success_rate = "unknown"
                    except:
                        curriculum_distance = "unknown"
                        curriculum_success_rate = "unknown"
                    
                    # Log detailed episode data to CSV
                    with open(self.csv_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            self.total_episodes,
                            goal_reached,
                            episode_reward,
                            episode_length,
                            curriculum_distance,
                            curriculum_success_rate,
                            datetime.now().isoformat()
                        ])
                    
                    # Export GPX if goal was reached and trajectory is available
                    if goal_reached:
                        self.episode += 1
                        
                        # Access the underlying environment directly to get trajectory
                        try:
                            # Get the actual environment (unwrap from SB3 wrappers)
                            env = self.training_env
                            if hasattr(env, 'envs') and len(env.envs) > 0:
                                # Vector environment - get first env
                                actual_env = env.envs[0]
                                # Unwrap further if needed
                                while hasattr(actual_env, 'env'):
                                    actual_env = actual_env.env
                                
                                # Get trajectory from actual environment
                                if hasattr(actual_env, 'trajectory') and actual_env.trajectory:
                                    traj = [(float(pos[0]), float(pos[1])) for pos in actual_env.trajectory]
                                    
                                    # Create GPX
                                    gpx = gpxpy.gpx.GPX()
                                    gpx_track = gpxpy.gpx.GPXTrack()
                                    gpx.tracks.append(gpx_track)
                                    gpx_segment = gpxpy.gpx.GPXTrackSegment()
                                    gpx_track.segments.append(gpx_segment)

                                    for lat, lon in traj:
                                        gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon, elevation=0))

                                    out_file = TRAJ_DIR / f"successful_episode_{self.episode}.gpx"
                                    with open(out_file, "w") as f:
                                        f.write(gpx.to_xml())
                                    print(f"💾 Trajectory saved: {out_file}")
                                    self.progress_logger.info(f"GPX exported to {out_file}")
                        except Exception as e:
                            # Fallback to info trajectory if available
                            if "trajectory" in info:
                                traj = info["trajectory"]

                                gpx = gpxpy.gpx.GPX()
                                gpx_track = gpxpy.gpx.GPXTrack()
                                gpx.tracks.append(gpx_track)
                                gpx_segment = gpxpy.gpx.GPXTrackSegment()
                                gpx_track.segments.append(gpx_segment)

                                for lat, lon in traj:
                                    gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon, elevation=0))

                                out_file = TRAJ_DIR / f"successful_episode_{self.episode}.gpx"
                                with open(out_file, "w") as f:
                                    f.write(gpx.to_xml())
                                print(f"💾 Trajectory saved: {out_file}")
                                self.progress_logger.info(f"GPX exported to {out_file}")
        
        return True
        
    def _on_training_end(self) -> None:
        """Close progress bar and log training completion"""
        if self.progress_bar:
            self.progress_bar.close()
        
        final_success_rate = (self.successful_episodes / max(1, self.total_episodes)) * 100
        self.progress_logger.info(f"Training completed: {self.successful_episodes}/{self.total_episodes} episodes successful ({final_success_rate:.1f}%)")
        print(f"\n🏁 Training completed: {self.successful_episodes}/{self.total_episodes} episodes successful ({final_success_rate:.1f}%)")
        return True


def make_env():
    """Create environment with proper step limit enforcement"""
    
    env = RealisticHikingEnv(
        processed_data_dir="data/processed",
        patch_size=64,
        max_steps=None,  # Dynamic step limits calculated by environment
        auto_save_gpx=True,
        rng_seed=None,
        curriculum_learning=True,      # Enable curriculum learning
        start_distance_meters=8.0,     # Start easier at 8m - give agent taste of success!
        include_goal_in_obs=True,      # Give agent goal direction info
    )
    
    log_dir = f"./logs/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)  # log episode rewards
    return env


def main():
    # -------------------------------
    # Create environment
    # -------------------------------
    env = DummyVecEnv([make_env])

    # -------------------------------
    # PPO Hyperparameters (Adjusted for Sparse Rewards)
    # -------------------------------
    model = PPO(
        "MultiInputPolicy",  # Required for Dict observation space
        env,
        verbose=1,
        learning_rate=3e-5,  # Even lower for stability with longer episodes
        n_steps=4096,        # Much longer rollouts for sparse rewards
        batch_size=128,      # Larger batches for better gradient estimates
        n_epochs=3,          # Fewer epochs to prevent overfitting
        gamma=0.995,         # Higher discount for longer episodes
        clip_range=0.2,
        gae_lambda=0.95,
        ent_coef=0.35,       # INCREASED entropy for breaking wandering patterns
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(OUTPUT_DIR / "tensorboard"),
    )

    # -------------------------------
    # Training loop with progress tracking
    # -------------------------------
    total_timesteps = 2_000_000  # Continue long training for curriculum progression
    callback = GPXExportCallback(verbose=1)

    print("🔧 Starting RECOVERY training to fix policy collapse...")
    print("✅ IMPROVED REWARD SHAPING:")
    print("- STRONGER progress bonuses (up to 5 points, lower threshold)")
    print("- REDUCED backtracking penalties to allow exploration")
    print("- INCREASED step limits (15-20 steps/m) for learning navigation")
    print("- ADDED proximity bonus for getting closer to goal")
    print()
    print("🧠 LEARNING RECOVERY:")
    print("- INCREASED exploration (ent_coef=0.35) to break wandering patterns")
    print("- Curriculum progression with sliding window evaluation")
    print("- Time-based advancement to prevent stalls")
    print(f"- Training for {total_timesteps:,} timesteps with enhanced exploration")
    
    model.learn(
        total_timesteps=total_timesteps, 
        callback=callback,
        tb_log_name=f"Fixed_Reward_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = CHECKPOINT_DIR / f"fixed_reward_model_{timestamp}.zip"
    model.save(model_path)
    print(f"[Trainer] Saved FIXED REWARD model to {model_path}")
    print("This model uses progress-based rewards and should not get stuck resting!")


if __name__ == "__main__":
    main()
