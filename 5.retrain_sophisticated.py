# retrain_sophisticated.py
"""
Retrain the agent with enhanced reward structure that encourages sophisticated navigation
instead of simple single-direction movement. This addresses the "policy collapse" issue
where the agent only learned to use Action 0.
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
import time
from tqdm import tqdm

OUTPUT_DIR = Path("outputs")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

class SophisticatedNavigationCallback(BaseCallback):
    """
    Callback that tracks action diversity and navigation sophistication
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode = 0
        self.successful_episodes = 0
        self.action_counts = np.zeros(9)
        self.last_log_time = time.time()
        
    def _on_step(self) -> bool:
        # Track actions taken
        infos = self.locals.get("infos", [])
        for info in infos:
            if "result" in info:
                self.episode += 1
                
                # Count success
                if info.get("reached_goal", False):
                    self.successful_episodes += 1
                
                # Log every 50 episodes
                if self.episode % 50 == 0:
                    success_rate = self.successful_episodes / max(1, self.episode)
                    current_time = time.time()
                    elapsed = current_time - self.last_log_time
                    
                    print(f"📊 Episode {self.episode}: Success rate: {success_rate:.1%} ({self.successful_episodes}/{self.episode})")
                    
                    # Track action diversity from environment
                    env = self.training_env.envs[0]
                    if hasattr(env, 'recent_actions') and len(env.recent_actions) > 0:
                        unique_recent = len(set(env.recent_actions[-10:]))
                        print(f"🎯 Action diversity: {unique_recent}/8 unique actions in last 10 steps")
                    
                    self.last_log_time = current_time
        
        return True

def main():
    print("🔧 SOPHISTICATED NAVIGATION RETRAINING")
    print("✅ Enhanced reward structure with:")
    print("  - Action diversity bonuses") 
    print("  - Anti-stuck penalties")
    print("  - Sophisticated pathfinding rewards")
    print("  - Higher exploration to break policy collapse")
    print()

    # Create environment with curriculum learning
    env = RealisticHikingEnv(
        curriculum_learning=True,
        auto_save_gpx=False,
        include_goal_in_obs=False,
    )
    
    # First episode to see initial setup
    obs, info = env.reset()
    print(f"🏔️ Trail info: Starting {env.start_distance_meters:.0f}m from summit")
    print(f"🎯 Goal position: {env.goal}")
    print(f"🚶 Start position: {env.current_pos}")
    print(f"📏 Distance: {np.linalg.norm(env.current_pos - env.goal) * env.cell_size:.1f}m")
    print()

    # Wrap for stable-baselines3
    env = Monitor(env, str(OUTPUT_DIR / "monitor"))
    env = DummyVecEnv([lambda: env])

    # Enhanced PPO with VERY HIGH exploration
    print("🧠 PPO Configuration:")
    print("  - VERY HIGH entropy (0.5) to break policy collapse")
    print("  - Lower learning rate for stability") 
    print("  - Longer rollouts for sparse rewards")
    print("  - Higher gamma for long-term planning")
    
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=1e-5,   # LOWER learning rate for stability
        n_steps=8192,         # LONGER rollouts for complex navigation
        batch_size=256,       # LARGER batches
        n_epochs=4,           # More epochs for learning
        gamma=0.998,          # HIGHER discount for very long episodes
        clip_range=0.3,       # Slightly higher clip range
        gae_lambda=0.95,
        ent_coef=0.5,         # VERY HIGH entropy to force exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(OUTPUT_DIR / "tensorboard"),
    )

    # Training
    total_timesteps = 3_000_000  # Longer training for sophisticated behavior
    callback = SophisticatedNavigationCallback(verbose=1)
    
    print(f"🎯 Training for {total_timesteps:,} timesteps...")
    print(f"📝 Logging to {OUTPUT_DIR / 'tensorboard'}")
    print()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=f"Sophisticated_Nav_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Save final model
        model_path = CHECKPOINT_DIR / f"sophisticated_navigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        model.save(str(model_path))
        print(f"✅ Model saved to {model_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        model_path = CHECKPOINT_DIR / f"sophisticated_navigation_interrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        model.save(str(model_path))
        print(f"💾 Progress saved to {model_path}")

if __name__ == "__main__":
    main()
