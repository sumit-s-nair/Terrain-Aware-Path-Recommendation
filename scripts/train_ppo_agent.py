import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import gymnasium as gym
from terrain_nav_env import TerrainNavEnv

def create_terrain_env():
    """Create a single terrain navigation environment."""
    # Create a more complex elevation map for training
    map_size = 100
    elevation_map = np.random.rand(map_size, map_size) * 50
    
    # Add terrain features cause the world isn't flat
    # Mountain ranges
    for _ in range(3):
        center_r = np.random.randint(20, 80)
        center_c = np.random.randint(20, 80)
        for r in range(map_size):
            for c in range(map_size):
                dist = np.sqrt((r - center_r)**2 + (c - center_c)**2)
                if dist < 15:
                    elevation_map[r, c] += 80 * (1 - dist/15)
    
    # Valleys and rivers
    for _ in range(2):
        start_r = np.random.randint(0, map_size)
        start_c = np.random.randint(0, map_size)
        for i in range(30):
            r = max(0, min(map_size-1, start_r + i + np.random.randint(-2, 3)))
            c = max(0, min(map_size-1, start_c + np.random.randint(-2, 3)))
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < map_size and 0 <= cc < map_size:
                        elevation_map[rr, cc] -= 20
    
    # Random start and goal positions
    start_pos = (np.random.randint(5, 15), np.random.randint(5, 15))
    goal_pos = (np.random.randint(85, 95), np.random.randint(85, 95))
    
    env = TerrainNavEnv(
        elevation_map=elevation_map,
        goal_coord=goal_pos,
        start_coord=start_pos,
        max_steps=1000,
        slope_threshold=15.0
    )
    
    return env

def train_ppo_agent():
    """Train a PPO agent on the terrain navigation environment."""
    
    print("Starting training of PPO agent for terrain navigation...")
    
    # Create directories for saving cause ill forget
    model_dir = Path("models")
    log_dir = Path("logs")
    model_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    print("Creating vectorized environment with 4 parallel instances...")
    
    # Create vectorized environment with 4 parallel instances
    vec_env = make_vec_env(
        create_terrain_env,
        n_envs=4,
        vec_env_cls=DummyVecEnv
    )
    
    print("Initializing PPO agent...")
    
    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=str(log_dir),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cpu"
    )
    
    # Create evaluation environment
    eval_env = create_terrain_env()
    
    # Callbacks for monitoring training
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="ppo_terrain_nav"
    )
    
    print("Starting training for 100,000 timesteps...")
    
    # Train the agent
    model.learn(
        total_timesteps=10000000,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name="PPO_TerrainNav",
        progress_bar=True
    )
    
    print("Training completed! Saving final model...")
    
    # Save the final model
    model.save(model_dir / "ppo_terrain_nav_final")
    
    print(f"Model saved to {model_dir / 'ppo_terrain_nav_final'}")
    
    # Plot training curve
    plot_training_curve(log_dir)
    
    # Test the trained agent
    test_trained_agent(model, create_terrain_env())
    
    return model

def plot_training_curve(log_dir: Path):
    """Plot the training reward curve from TensorBoard logs."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        print("Plotting training curve...")
        
        # Find the TensorBoard log file
        tb_files = list(log_dir.glob("**/events.out.tfevents.*"))
        if not tb_files:
            print("No TensorBoard log files found.")
            return
        
        # Load the most recent log file
        event_file = str(tb_files[-1])
        event_accumulator = EventAccumulator(event_file)
        event_accumulator.Reload()
        
        # Extract training rewards
        if 'rollout/ep_rew_mean' in event_accumulator.Tags()['scalars']:
            rewards = event_accumulator.Scalars('rollout/ep_rew_mean')
            steps = [x.step for x in rewards]
            values = [x.value for x in rewards]
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(steps, values)
            plt.title('Training Reward Curve')
            plt.xlabel('Timesteps')
            plt.ylabel('Episode Reward')
            plt.grid(True)
        
        # Extract episode lengths
        if 'rollout/ep_len_mean' in event_accumulator.Tags()['scalars']:
            lengths = event_accumulator.Scalars('rollout/ep_len_mean')
            steps = [x.step for x in lengths]
            values = [x.value for x in lengths]
            
            plt.subplot(1, 2, 2)
            plt.plot(steps, values)
            plt.title('Episode Length')
            plt.xlabel('Timesteps')
            plt.ylabel('Episode Length')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Training curves saved as 'training_curves.png'")
        
    except ImportError:
        print("TensorBoard not available for plotting. Install with: pip install tensorboard")
    except Exception as e:
        print(f"Error plotting training curve: {e}")

def test_trained_agent(model, env, num_episodes=3):
    """Test the trained agent on the environment."""
    print(f"\nTesting trained agent for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs, info = env.reset()  # Handle tuple return from reset()
        total_reward = 0
        steps = 0
        done = False
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)  # Handle 5 values from step()
            total_reward += reward
            steps += 1
        
        print(f"Episode {episode + 1}: Steps: {steps}, Total Reward: {total_reward:.2f}, "
              f"Goal Reached: {info.get('goal_reached', False)}")
        
        if episode == 0:  # Render first episode
            env.render()

if __name__ == "__main__":
    # Train the agent to teach it to navigate the terrain
    print("Starting training of PPO agent for terrain navigation...")
    trained_model = train_ppo_agent()
    
    print("\nTraining complete! You can now:")
    print("1. View TensorBoard logs: tensorboard --logdir logs")
    print("2. Load the model: model = PPO.load('models/ppo_terrain_nav_final')")
    print("3. Test on new environments using the saved model")
