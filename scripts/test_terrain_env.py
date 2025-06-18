import numpy as np
import matplotlib.pyplot as plt
from terrain_nav_env import TerrainNavEnv

def test_terrain_env():
    """Test the terrain navigation environment with improved reward system."""
    
    # Create a simple test elevation map
    elevation_map = np.random.rand(50, 50) * 100  # Random elevations 0-100m
    
    # Add some terrain features
    # Mountain in center
    center_r, center_c = 25, 25
    for r in range(50):
        for c in range(50):
            dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
            if dist < 10:
                elevation_map[r, c] += 50 * (1 - dist/10)
    
    # Valley
    for r in range(10, 20):
        for c in range(35, 45):
            elevation_map[r, c] -= 30
    
    # Set goal and start positions
    start_pos = (5, 5)
    goal_pos = (45, 45)
    
    # Create environment with custom slope threshold
    env = TerrainNavEnv(
        elevation_map=elevation_map,
        goal_coord=goal_pos,
        start_coord=start_pos,
        max_steps=500,
        slope_threshold=12.0  # Custom slope threshold
    )
    
    # Test episode
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    total_reward = 0
    step_count = 0
    
    # Track reward components for analysis
    reward_history = []
    distance_history = []
    visited_cells_history = []
    
    # Run test episode
    while not env.done and step_count < 100:  # Limit for demo
        action = env.action_space.sample()  # Random action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Track metrics
        reward_history.append(reward)
        distance_history.append(info['distance_to_goal'])
        visited_cells_history.append(info['visited_cells'])
        
        if step_count % 20 == 0:
            print(f"Step {step_count}: Position {env.current_pos}, "
                  f"Reward: {reward:.2f}, Distance: {info['distance_to_goal']:.2f}, "
                  f"Visited: {info['visited_cells']}")
    
    print(f"\nEpisode finished!")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final distance to goal: {env._distance_to_goal():.2f}")
    print(f"Goal reached: {env.current_pos == goal_pos}")
    
    # Render final state
    env.render()
    
    # Plot reward and distance over time
    plot_training_metrics(reward_history, distance_history, visited_cells_history)
    
    return env

def plot_training_metrics(rewards, distances, visited_cells):
    """Plot training metrics to understand reward structure."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Reward over time
    axes[0, 0].plot(rewards)
    axes[0, 0].set_title('Reward per Step')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Distance to goal over time
    axes[0, 1].plot(distances)
    axes[0, 1].set_title('Distance to Goal')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Distance')
    axes[0, 1].grid(True)
    
    # Cumulative reward
    cumulative_rewards = np.cumsum(rewards)
    axes[1, 0].plot(cumulative_rewards)
    axes[1, 0].set_title('Cumulative Reward')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Cumulative Reward')
    axes[1, 0].grid(True)
    
    # Visited cells over time
    axes[1, 1].plot(visited_cells)
    axes[1, 1].set_title('Visited Cells')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Number of Visited Cells')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_terrain_env()
