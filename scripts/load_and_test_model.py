import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from terrain_nav_env import TerrainNavEnv
import matplotlib.pyplot as plt

def load_and_test_model():
    """Load trained model and test on new environments."""
    
    model_path = Path("models/ppo_terrain_nav_final")
    
    if not model_path.exists():
        print(f"Model not found at {model_path}. Please train the model first.")
        return
    
    print("Loading trained model...")
    model = PPO.load(model_path)
    
    # Create a test environment
    test_env = create_test_environment()
    
    print("Testing on new environment...")
    obs, info = test_env.reset()  # Handle tuple return
    total_reward = 0
    steps = 0
    path = [test_env.current_pos]
    terminated = False
    truncated = False
    
    while not (terminated or truncated) and steps < 1000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)  # Handle 5 values
        total_reward += reward
        steps += 1
        path.append(test_env.current_pos)
        
        if steps % 50 == 0:
            print(f"Step {steps}: Position {test_env.current_pos}, "
                  f"Distance to goal: {info['distance_to_goal']:.2f}")
    
    print(f"\nTest Results:")
    print(f"Steps taken: {steps}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Goal reached: {info.get('goal_reached', False)}")
    print(f"Final distance to goal: {test_env._distance_to_goal():.2f}")
    
    # Visualize the path taken
    visualize_agent_path(test_env, path)
    
    return model, test_env, path

def create_test_environment():
    """Create a test environment, optionally using real elevation data."""
    try:
        # Try to load real elevation data if available we dont have rn so we do test env
        processed_folder = Path("../data/processed")
        npy_files = list(processed_folder.glob("*_upscaled.npy"))
        
        if npy_files:
            print(f"Using real elevation data: {npy_files[0].name}")
            elevation_map = np.load(npy_files[0])
            
            # Handle NaN values
            elevation_map = np.nan_to_num(elevation_map, nan=0.0)
            
            # Resize if too large
            if elevation_map.shape[0] > 200 or elevation_map.shape[1] > 200:
                from scipy.ndimage import zoom
                scale_factor = 200 / max(elevation_map.shape)
                elevation_map = zoom(elevation_map, scale_factor, order=1)
            
            rows, cols = elevation_map.shape
            start_pos = (10, 10)
            goal_pos = (rows - 10, cols - 10)
            
        else:
            raise FileNotFoundError("No elevation data found")
            
    except:
        print("Using synthetic elevation data for testing...")
        # Create synthetic test environment
        elevation_map = np.random.rand(80, 80) * 100
        
        # Add some challenging terrain
        center_r, center_c = 40, 40
        for r in range(80):
            for c in range(80):
                dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
                if dist < 20:
                    elevation_map[r, c] += 150 * (1 - dist/20)
        
        start_pos = (5, 5)
        goal_pos = (75, 75)
    
    env = TerrainNavEnv(
        elevation_map=elevation_map,
        goal_coord=goal_pos,
        start_coord=start_pos,
        max_steps=1000
    )
    
    return env

def visualize_agent_path(env, path):
    """Visualize the path taken by the agent."""
    plt.figure(figsize=(12, 10))
    
    # Show elevation map
    plt.imshow(env.elevation_map, cmap='terrain', origin='upper')
    plt.colorbar(label='Elevation (m)')
    
    # Plot path
    if len(path) > 1:
        path_rows = [pos[0] for pos in path]
        path_cols = [pos[1] for pos in path]
        plt.plot(path_cols, path_rows, 'r-', linewidth=2, alpha=0.8, label='Agent Path')
    
    # Mark start, current position, and goal
    start_pos = path[0] if path else env.start_coord
    plt.plot(start_pos[1], start_pos[0], 'bs', markersize=10, label='Start')
    plt.plot(env.current_pos[1], env.current_pos[0], 'ro', markersize=10, label='Final Position')
    plt.plot(env.goal_coord[1], env.goal_coord[0], 'g*', markersize=15, label='Goal')
    
    plt.title('Trained Agent Path on Terrain')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('agent_path.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    model, env, path = load_and_test_model()
