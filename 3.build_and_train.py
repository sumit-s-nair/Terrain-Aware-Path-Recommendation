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
from tqdm import tqdm
import time

from datetime import datetime
import gpxpy
import gpxpy.gpx


OUTPUT_DIR = Path("outputs")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
TRAJ_DIR = OUTPUT_DIR / "trajectories"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
TRAJ_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------
# Callback to export GPX on success
# -------------------------------
class GPXExportCallback(BaseCallback):
    """Enhanced callback with progress bar and GPX export for successful episodes"""
    
    def __init__(self, save_freq=1, verbose=0):
        super().__init__(verbose)
        self.episode = 0
        self.successful_episodes = 0
        self.progress_bar = None
        self.last_update_time = time.time()
        
    def _on_training_start(self) -> None:
        """Initialize progress bar at training start"""
        total_timesteps = self.locals.get('total_timesteps', 1000000)
        self.progress_bar = tqdm(
            total=total_timesteps,
            desc="🏔️ Training Hiking Agent",
            unit="steps", 
            ncols=120,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Success: {postfix}"
        )
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
                
                # Simple success tracking - just use our successful episodes count
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
                            f"Total successes: {self.successful_episodes}"
                        )
                    else:
                        # Fallback to simple tracking
                        self.progress_bar.set_postfix_str(f"Success: {self.successful_episodes} episodes")
                except Exception as e:
                    self.progress_bar.set_postfix_str(f"Success: {self.successful_episodes} episodes")
                
            self.last_update_time = current_time
        
        # Check if episode ended and goal was reached
        if self.locals.get("dones") is not None and np.any(self.locals["dones"]):
            infos = self.locals["infos"]
            for info in infos:
                # Print episode completion info
                if "reached_goal" in info:
                    episode_reward = info.get("episode", {}).get("r", "unknown")
                    episode_length = info.get("episode", {}).get("l", "unknown")
                    
                    if info.get("reached_goal", False):
                        print(f"\n🎯 SUCCESS! Episode reward: {episode_reward}, Length: {episode_length} steps")
                    else:
                        print(f"\n❌ Episode ended. Reward: {episode_reward}, Length: {episode_length} steps")
                
                # Only export GPX if the goal was actually reached
                if info.get("reached_goal", False) and "trajectory" in info:
                    self.episode += 1
                    self.successful_episodes += 1
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
                    if self.verbose > 0:
                        print(f"[GPXExport] Saved successful trajectory to {out_file}")
        return True
        
    def _on_training_end(self) -> None:
        """Close progress bar when training ends"""
        if self.progress_bar:
            self.progress_bar.close()
        return True


def make_env():
    """Create environment with FIXED reward structure that rewards progress, not proximity"""
    
    class FixedRealisticHikingEnv(RealisticHikingEnv):
        """Fixed version that rewards progress toward goal instead of just being close"""
        
        def step(self, action: int):
            self.step_count += 1
            prev = self.current_pos.copy()

            # 8-connected movement + rest
            if action == 8:  # rest
                intended = self.current_pos.copy()
                self.energy = min(self.ENERGY_MAX, self.energy + 2.0)
                result = "rest"
            else:
                dirs = {
                    0: (-1, 0), 1: (-1, 1), 2: (0, 1), 3: (1, 1),
                    4: (1, 0), 5: (1, -1), 6: (0, -1), 7: (-1, -1),
                }
                action_idx = int(action.item()) if hasattr(action, 'item') else int(action)
                d = np.array(dirs[action_idx], dtype=np.float32)
                intended = self.current_pos + d

                valid, reason = self._movement_is_valid(self.current_pos, intended)
                if valid:
                    nxt, slip_state = self._apply_slip_if_needed(intended)
                    self.current_pos = nxt
                    result = slip_state
                else:
                    result = f"blocked_{reason}"

            self.trajectory.append(self.current_pos.copy())

            # ---- FIXED REWARD STRUCTURE ----
            reached = self._goal_reached()
            reward = 0.0
            
            # Goal reach bonus - huge reward for actually reaching goal
            if reached:
                reward += 1000.0
                if self.curriculum_learning:
                    self.curriculum_successes += 1
                
            # Death penalty
            if self.health <= 0.0:
                reward -= 100.0
                
            # MAIN CHANGE: Progress-based reward instead of distance-based
            prev_dist = np.linalg.norm(prev - self.goal) * self.cell_size
            cur_dist = np.linalg.norm(self.current_pos - self.goal) * self.cell_size
            progress = prev_dist - cur_dist  # Positive = moving toward goal
            
            # Reward ONLY progress, not just being close
            if progress > 0:
                reward += progress * 10.0  # Reward for moving toward goal
            elif progress < 0:
                reward += progress * 5.0   # Penalty for moving away from goal
            # No reward/penalty for staying still (progress = 0)
            
            # Small base reward for staying alive
            reward += 0.1
            
            # Small time penalty to encourage efficiency
            reward -= 0.01
            
            # Light penalties for risky moves
            if "slip" in result:
                reward -= 1.0
            if "blocked" in result:
                reward -= 0.5
                
            # STRONG penalty for resting when not at goal
            if action == 8 and not reached:
                reward -= 2.0  # Discourage resting
                
            terminated = reached or self.health <= 0.0
            truncated = False  # No step limits - let agent take unlimited time!
            
            return self._obs(), reward, terminated, truncated, {"result": result, "reached_goal": reached}
    
    env = FixedRealisticHikingEnv(
        processed_data_dir="data/processed",
        patch_size=64,
        max_steps=None,  # REMOVED step limits - let agent take as long as needed!
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
    # PPO Hyperparameters
    # -------------------------------
    model = PPO(
        "MultiInputPolicy",  # Required for Dict observation space
        env,
        verbose=1,
        learning_rate=1e-4,  # Reduced to prevent premature convergence
        n_steps=1024,  # Shorter rollout batches for faster feedback
        batch_size=64,
        n_epochs=4,    # Reduced to prevent overfitting
        gamma=0.99,
        clip_range=0.2,
        gae_lambda=0.95,
        ent_coef=0.15,  # Very high entropy for 10m training to ensure exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(OUTPUT_DIR / "tensorboard"),
    )

    # -------------------------------
    # Training loop with progress tracking
    # -------------------------------
    total_timesteps = 1_000_000  # Start with shorter training for validation
    callback = GPXExportCallback(verbose=1)

    print("🏔️ Starting ENHANCED training with FIXED reward structure...")
    print("- Rewards progress toward goal, not just proximity")
    print("- Penalizes resting when not at goal")  
    print("- Uses curriculum learning starting at 8m distance (achievable start)")
    print("- NO STEP LIMITS - agent can take as long as needed to reach goal!")
    print("- Episodes end only on: goal reached, health depleted, or energy exhausted")
    print("- AGGRESSIVE curriculum progression (75% success, 2x DOUBLING)")
    print("- Real-time progress bar with success tracking")
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
