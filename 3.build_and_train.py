# build_and_train.py
"""
Train PPO agent in physics-based hiking environment.
"""

import os
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from physics_hiking_env import RealisticHikingEnv

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
    def __init__(self, save_freq=1, verbose=0):
        super().__init__(verbose)
        self.episode = 0

    def _on_step(self) -> bool:
        # SB3 triggers after each step, but we want end of episode
        if self.locals.get("dones") is not None and np.any(self.locals["dones"]):
            infos = self.locals["infos"]
            for info in infos:
                # Only export GPX if the goal was actually reached
                if info.get("reached_goal", False) and "trajectory" in info:
                    self.episode += 1
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
                        print(f"[GPXExport] Goal reached! Saved successful trajectory to {out_file}")
        return True


def make_env():
    env = RealisticHikingEnv(
        processed_data_dir="data/processed",
        patch_size=64,
        max_steps=None,  # No step limit - episodes end only on goal/health<=0 (energy disabled)
        auto_save_gpx=True,
        rng_seed=None,
        curriculum_learning=True,      # Enable curriculum learning
        start_distance_meters=300.0,   # Start 300m from goal
        include_goal_in_obs=True,      # Give agent goal direction info
    )
    env = Monitor(env)  # log episode rewards
    return env


def main():
    # -------------------------------
    # Create environment
    # -------------------------------
    env = DummyVecEnv([make_env])

    # -------------------------------
    # PPO Hyperparameters (tuned for physics env)
    # -------------------------------
    model = PPO(
        "MultiInputPolicy",  # Required for Dict observation space
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,  # Shorter rollout batches for faster feedback
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        gae_lambda=0.95,
        ent_coef=0.05,  # Increased exploration entropy
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(OUTPUT_DIR / "tensorboard"),
    )

    # -------------------------------
    # Training loop
    # -------------------------------
    total_timesteps = 1_000_000
    callback = GPXExportCallback(verbose=1)

    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = CHECKPOINT_DIR / f"ppo_hiker_{timestamp}.zip"
    model.save(model_path)
    print(f"[Trainer] Saved model to {model_path}")


if __name__ == "__main__":
    main()
