#!/usr/bin/env python3
"""
Comprehensive test script to evaluate trained PPO agent performance with fixed reward structure.
Provides detailed diagnostics and behavior analysis to detect stuck agents and policy collapse.
"""

from physics_hiking_env import RealisticHikingEnv
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob


class FixedRealisticHikingEnv(RealisticHikingEnv):
    """Fixed version for testing that uses progress-based rewards"""
    
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
        
        if reached:
            reward += 1000.0
            if self.curriculum_learning:
                self.curriculum_successes += 1
            
        if self.health <= 0.0:
            reward -= 100.0
            
        # Progress-based reward
        prev_dist = np.linalg.norm(prev - self.goal) * self.cell_size
        cur_dist = np.linalg.norm(self.current_pos - self.goal) * self.cell_size
        progress = prev_dist - cur_dist
        
        if progress > 0:
            reward += progress * 10.0
        elif progress < 0:
            reward += progress * 5.0
        
        reward += 0.1  # alive bonus
        reward -= 0.01  # time penalty
        
        if "slip" in result:
            reward -= 1.0
        if "blocked" in result:
            reward -= 0.5
        if action == 8 and not reached:
            reward -= 2.0  # discourage resting
            
        terminated = reached or self.health <= 0.0
        truncated = self.step_count >= self.max_steps
        
        return self._obs(), reward, terminated, truncated, {"result": result}


def find_latest_model():
    """Find the most recent model checkpoint."""
    checkpoint_dir = Path("outputs/checkpoints")
    if not checkpoint_dir.exists():
        print("No checkpoint directory found!")
        return None
    
    model_files = glob.glob(str(checkpoint_dir / "*.zip"))
    if not model_files:
        print("No model files found in checkpoints!")
        return None
    
    # Sort by modification time, get latest
    latest_model = max(model_files, key=lambda x: Path(x).stat().st_mtime)
    return latest_model


def evaluate_agent_performance(model_path, num_episodes=3, max_steps_per_episode=500):
    """Run multiple test episodes with comprehensive analysis."""
    
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)
    
    # Test with different start distances to evaluate robustness
    test_configs = [
        {"start_distance": 50.0, "name": "Training Distance (50m)"},
        {"start_distance": 100.0, "name": "Close Start (100m)"},
        {"start_distance": 200.0, "name": "Medium Start (200m)"},
        {"start_distance": 500.0, "name": "Far Start (500m)"}
    ]
    
    all_results = []
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"{'='*60}")
        
        env = FixedRealisticHikingEnv(
            processed_data_dir="data/processed",
            patch_size=64,
            max_steps=max_steps_per_episode,
            auto_save_gpx=False,
            rng_seed=42,
            curriculum_learning=True,
            start_distance_meters=config["start_distance"],
            include_goal_in_obs=True,
        )
        
        episode_results = []
        
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1} ---")
            
            obs, info = env.reset()
            total_reward = 0
            total_movement = 0
            action_counts = {}
            positions = [env.current_pos.copy()]
            
            initial_distance = np.linalg.norm(env.current_pos - env.goal) * env.cell_size
            best_distance = initial_distance
            
            print(f"Start: [{env.current_pos[0]:.0f}, {env.current_pos[1]:.0f}] -> Goal: [{env.goal[0]:.0f}, {env.goal[1]:.0f}]")
            print(f"Initial distance: {initial_distance:.1f}m")
            
            for step in range(max_steps_per_episode):
                action, _ = model.predict(obs, deterministic=True)
                action_int = int(action)
                action_counts[action_int] = action_counts.get(action_int, 0) + 1
                
                prev_pos = env.current_pos.copy()
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Calculate movement and progress
                movement = np.linalg.norm(env.current_pos - prev_pos) * env.cell_size
                total_movement += movement
                positions.append(env.current_pos.copy())
                
                current_distance = np.linalg.norm(env.current_pos - env.goal) * env.cell_size
                if current_distance < best_distance:
                    best_distance = current_distance
                
                total_reward += reward
                
                # Print progress periodically
                if step % 100 == 0 or step < 10:
                    print(f"Step {step}: Action={action_int}, Dist={current_distance:.1f}m, Movement={movement:.2f}m, Reward={reward:.2f}")
                
                if terminated:
                    if current_distance < 5.0:
                        print(f"🎉 SUCCESS! Goal reached at step {step}!")
                        break
                    else:
                        print(f"Episode terminated at step {step}")
                        break
                
                if truncated:
                    print(f"Episode truncated at step {step}")
                    break
            
            final_distance = current_distance
            progress_percent = ((initial_distance - final_distance) / initial_distance) * 100
            
            # Action diversity analysis
            action_diversity = len(action_counts)
            most_common_action = max(action_counts, key=action_counts.get)
            action_percentage = (action_counts[most_common_action] / sum(action_counts.values())) * 100
            
            episode_result = {
                'episode': episode + 1,
                'initial_distance': initial_distance,
                'final_distance': final_distance,
                'best_distance': best_distance,
                'progress_percent': progress_percent,
                'total_movement': total_movement,
                'total_reward': total_reward,
                'steps': step + 1,
                'action_diversity': action_diversity,
                'action_counts': action_counts,
                'most_common_action': most_common_action,
                'action_percentage': action_percentage,
                'success': final_distance < 5.0
            }
            
            episode_results.append(episode_result)
            
            print(f"Result: {progress_percent:.1f}% progress, {total_movement:.1f}m movement")
            print(f"Actions: {action_diversity} different, most common: {most_common_action} ({action_percentage:.1f}%)")
            
            # Behavior diagnosis
            if episode_result['success']:
                print("✅ SUCCESS: Agent reached the goal!")
            elif total_movement < 5.0:
                print("❌ STUCK: Agent barely moved")
            elif action_diversity == 1:
                print("⚠️ POLICY COLLAPSE: Agent only uses one action")
            elif progress_percent > 50:
                print("✅ GOOD: Significant progress toward goal")
            elif progress_percent > 0:
                print("⚠️ PARTIAL: Some progress made")
            else:
                print("❌ POOR: Moving away from goal")
        
        # Calculate summary statistics
        config_summary = {
            'config_name': config['name'],
            'start_distance': config['start_distance'],
            'success_rate': sum(1 for r in episode_results if r['success']) / len(episode_results),
            'avg_progress': np.mean([r['progress_percent'] for r in episode_results]),
            'avg_movement': np.mean([r['total_movement'] for r in episode_results]),
            'avg_reward': np.mean([r['total_reward'] for r in episode_results]),
            'avg_steps': np.mean([r['steps'] for r in episode_results]),
            'avg_action_diversity': np.mean([r['action_diversity'] for r in episode_results]),
            'episodes': episode_results
        }
        
        all_results.append(config_summary)
        
        print(f"\n--- Summary for {config['name']} ---")
        print(f"Success Rate: {config_summary['success_rate']*100:.1f}%")
        print(f"Average Progress: {config_summary['avg_progress']:.1f}%")
        print(f"Average Movement: {config_summary['avg_movement']:.1f}m")
        print(f"Average Action Diversity: {config_summary['avg_action_diversity']:.1f}")
        print(f"Average Reward: {config_summary['avg_reward']:.1f}")
    
    return all_results


def create_performance_visualization(results):
    """Create visualization of agent performance across different configurations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Agent Performance Analysis', fontsize=16)
    
    config_names = [r['config_name'] for r in results]
    success_rates = [r['success_rate'] * 100 for r in results]
    avg_progress = [r['avg_progress'] for r in results]
    avg_movement = [r['avg_movement'] for r in results]
    avg_diversity = [r['avg_action_diversity'] for r in results]
    
    # Success Rate
    axes[0,0].bar(config_names, success_rates, color='green', alpha=0.7)
    axes[0,0].set_title('Success Rate (%)')
    axes[0,0].set_ylabel('Success Rate (%)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Progress
    axes[0,1].bar(config_names, avg_progress, color='blue', alpha=0.7)
    axes[0,1].set_title('Average Progress (%)')
    axes[0,1].set_ylabel('Progress (%)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Movement
    axes[1,0].bar(config_names, avg_movement, color='orange', alpha=0.7)
    axes[1,0].set_title('Average Movement (m)')
    axes[1,0].set_ylabel('Movement (m)')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Action Diversity
    axes[1,1].bar(config_names, avg_diversity, color='red', alpha=0.7)
    axes[1,1].set_title('Average Action Diversity')
    axes[1,1].set_ylabel('Number of Different Actions')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('agent_performance_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'agent_performance_analysis.png'")


def main():
    """Main function to test the latest model."""
    
    model_path = find_latest_model()
    if model_path is None:
        print("❌ No trained models found. Please run 3.build_and_train.py first.")
        return
    
    print(f"🤖 Testing latest model: {Path(model_path).name}")
    print("=" * 80)
    
    results = evaluate_agent_performance(model_path)
    
    # Overall assessment
    print(f"\n{'='*80}")
    print("OVERALL ASSESSMENT")
    print(f"{'='*80}")
    
    overall_success_rate = np.mean([r['success_rate'] for r in results])
    overall_progress = np.mean([r['avg_progress'] for r in results])
    overall_movement = np.mean([r['avg_movement'] for r in results])
    overall_diversity = np.mean([r['avg_action_diversity'] for r in results])
    
    print(f"Overall Success Rate: {overall_success_rate*100:.1f}%")
    print(f"Overall Average Progress: {overall_progress:.1f}%")
    print(f"Overall Average Movement: {overall_movement:.1f}m")
    print(f"Overall Action Diversity: {overall_diversity:.1f}")
    
    # Diagnosis
    if overall_success_rate > 0.5:
        print("✅ EXCELLENT: Agent successfully reaching goals!")
    elif overall_movement > 30 and overall_diversity > 2:
        print("✅ GOOD: Agent is exploring and making progress!")
    elif overall_movement > 10:
        print("⚠️ PARTIAL: Agent is moving but needs improvement")
    elif overall_diversity > 1:
        print("⚠️ FAIR: Agent trying different actions but limited movement")
    else:
        print("❌ POOR: Agent showing stuck behavior or policy collapse")
        print("   Consider retraining with different hyperparameters")
    
    create_performance_visualization(results)


if __name__ == "__main__":
    main()
