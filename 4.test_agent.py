#!/usr/bin/env python3
"""
Quick test to see how the agent performs.
"""

from physics_hiking_env import RealisticHikingEnv
from stable_baselines3 import PPO
import numpy as np

# Load model and environment
model = PPO.load("outputs/checkpoints/ppo_hiker_20250821_174300.zip")
env = RealisticHikingEnv()

# Run a test episode
obs, _ = env.reset()
positions = [env.current_pos.copy()]
rewards = []
step_count = 0
max_steps = 500000  # Safety limit

print(f"Starting position: {env.current_pos}")
print(f"Goal position: {env.goal}")
print(f"Distance to goal: {np.linalg.norm(env.goal - env.current_pos) * env.cell_size:.1f}m")
print("-" * 50)

while step_count < max_steps:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    positions.append(env.current_pos.copy())
    rewards.append(reward)
    step_count += 1
    
    if step_count % 100 == 0:
        dist = np.linalg.norm(env.goal - env.current_pos) * env.cell_size
        print(f"Step {step_count}: Position {env.current_pos}, Distance to goal: {dist:.1f}m, Reward: {reward:.2f}")
    
    if terminated or truncated:
        print(f"Episode ended at step {step_count}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        print(f"Final position: {env.current_pos}")
        print(f"Final distance to goal: {np.linalg.norm(env.goal - env.current_pos) * env.cell_size:.1f}m")
        print(f"Reached goal: {info.get('reached_goal', False)}")
        print(f"Result: {info.get('result', 'unknown')}")
        print(f"Energy: {info.get('energy', 0):.1f}, Health: {info.get('health', 0):.1f}")
        break

print(f"Total reward: {np.sum(rewards):.2f}")
print(f"Total steps: {len(positions)}")
