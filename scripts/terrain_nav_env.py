import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any

class TerrainNavEnv(gym.Env):
    """
    Custom OpenAI Gym environment for terrain navigation using elevation data.
    
    Agent navigates through terrain to reach a goal while considering elevation changes,
    slopes, and energy efficiency.
    """
    
    def __init__(self, elevation_map: np.ndarray, goal_coord: Tuple[int, int], 
                 start_coord: Optional[Tuple[int, int]] = None, max_steps: int = 1000,
                 slope_threshold: float = 15.0):
        """
        Initialize the terrain navigation environment.
        
        Args:
            elevation_map: 2D numpy array of elevation data
            goal_coord: (row, col) coordinate of the goal
            start_coord: (row, col) starting coordinate, random if None
            max_steps: Maximum steps per episode -- still debating if this should be a parameter or fixed
            slope_threshold: Maximum allowed slope before penalty
        """
        super(TerrainNavEnv, self).__init__()
        
        self.elevation_map = elevation_map
        self.map_height, self.map_width = elevation_map.shape
        self.goal_coord = goal_coord
        self.start_coord = start_coord
        self.max_steps = max_steps
        self.slope_threshold = slope_threshold  # Maximum allowed slope before penalty
        
        # Action space: 8 discrete movements (N, S, E, W, NE, NW, SE, SW)
        self.action_space = spaces.Discrete(8)
        
        # Movement deltas for each action
        self.action_deltas = {
            0: (-1, 0),   # North
            1: (1, 0),    # South
            2: (0, 1),    # East
            3: (0, -1),   # West
            4: (-1, 1),   # Northeast
            5: (-1, -1),  # Northwest
            6: (1, 1),    # Southeast
            7: (1, -1)    # Southwest
        }
        
        # Observation space: [current_row, current_col, current_elevation, 5x5_elevation_patch_flattened]
        obs_dim = 2 + 1 + 25  # position + elevation + 5x5 patch
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Environment state
        self.current_pos = None
        self.step_count = 0
        self.done = False
        self.visited_mask = None  # Track visited cells
        self.previous_distance = None  # Track distance to goal
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Set starting position
        if self.start_coord is None:
            # Random start position away from goal
            while True:
                row = np.random.randint(0, self.map_height)
                col = np.random.randint(0, self.map_width)
                if (row, col) != self.goal_coord and not np.isnan(self.elevation_map[row, col]):
                    self.current_pos = (row, col)
                    break
        else:
            self.current_pos = self.start_coord
            
        self.step_count = 0
        self.done = False
        
        # Initialize visited mask
        self.visited_mask = np.zeros((self.map_height, self.map_width), dtype=bool)
        self.visited_mask[self.current_pos[0], self.current_pos[1]] = True
        
        # Initialize distance tracking
        self.previous_distance = self._distance_to_goal()
        
        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        if self.done:
            raise ValueError("Episode has ended. Call reset() to start a new episode.")
        
        # Get action delta
        delta_row, delta_col = self.action_deltas[action]
        
        # Calculate new position
        new_row = self.current_pos[0] + delta_row
        new_col = self.current_pos[1] + delta_col
        
        # Check bounds and valid terrain
        if (0 <= new_row < self.map_height and 0 <= new_col < self.map_width and 
            not np.isnan(self.elevation_map[new_row, new_col])):
            
            # Valid move - update position and calculate reward
            old_pos = self.current_pos
            old_elevation = self.elevation_map[old_pos[0], old_pos[1]]
            new_elevation = self.elevation_map[new_row, new_col]
            
            self.current_pos = (new_row, new_col)
            
            # Calculate reward using improved method
            reward = self.compute_reward(old_pos, self.current_pos, old_elevation, new_elevation, action)
            
            # Mark cell as visited
            self.visited_mask[new_row, new_col] = True
            
        else:
            # Invalid move (out of bounds or NaN elevation)
            reward = -50.0  # Heavy penalty for invalid move
        
        self.step_count += 1
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        if self.current_pos == self.goal_coord:
            reward += 1000.0  # Goal reached bonus
            terminated = True
        elif self.step_count >= self.max_steps:
            reward -= 100.0  # Time penalty for not reaching goal
            truncated = True
        
        self.done = terminated or truncated
        
        # Update previous distance for next step
        self.previous_distance = self._distance_to_goal()
        
        info = {
            'step_count': self.step_count,
            'distance_to_goal': self.previous_distance,
            'current_elevation': self.elevation_map[self.current_pos[0], self.current_pos[1]],
            'visited_cells': np.sum(self.visited_mask),
            'goal_reached': self.current_pos == self.goal_coord
        }
        
        return self._get_observation(), reward, terminated, truncated, info

    def compute_reward(self, old_pos: Tuple[int, int], new_pos: Tuple[int, int], 
                      old_elevation: float, new_elevation: float, action: int) -> float:
        """
        Compute reward based on multiple factors for safe, efficient, non-repetitive paths.
        ig i covered all bases do add more if you want to improve it
        This function considers:
            1. Step penalty - encourage efficiency
            2. Distance reward - encourage moving towards goal
            3. Slope penalty - penalize steep elevation gains
            4. Revisiting penalty - discourage visiting already explored cells
            5. Exploration bonus - reward for visiting new cells
            6. Energy efficiency - prefer downhill or flat terrain

            yea wrote a lot so better read it
        
        Args:
            old_pos: Previous position (row, col)
            new_pos: New position (row, col)
            old_elevation: Elevation at previous position
            new_elevation: Elevation at new position
            action: Action taken (0-7)
            
        Returns:
            Calculated reward value
        """
        reward = 0.0
        
        # 1. Step penalty - encourage efficiency
        reward -= 0.1
        
        # 2. Distance reward - encourage moving towards goal
        current_distance = np.sqrt((new_pos[0] - self.goal_coord[0])**2 + 
                                 (new_pos[1] - self.goal_coord[1])**2)
        distance_improvement = self.previous_distance - current_distance
        reward += distance_improvement * 2.0  # Scale factor for distance reward
        
        # 3. Slope penalty - penalize steep elevation gains
        elevation_change = new_elevation - old_elevation
        
        # Calculate movement distance (diagonal vs cardinal)
        if action in [4, 5, 6, 7]:  # Diagonal moves
            movement_distance = np.sqrt(2)
        else:  # Cardinal moves
            movement_distance = 1.0
        
        # Calculate slope (rise over run)
        if movement_distance > 0:
            slope = abs(elevation_change) / movement_distance
            
            # Apply slope penalty if above threshold
            if slope > self.slope_threshold:
                slope_penalty = (slope - self.slope_threshold) * 0.5
                reward -= slope_penalty
            
            # Additional penalty for steep uphill climbs
            if elevation_change > 0 and slope > self.slope_threshold:
                reward -= elevation_change * 0.02
        
        # 4. Revisiting penalty - discourage visiting already explored cells
        if self.visited_mask[new_pos[0], new_pos[1]]:
            reward -= 5.0  # Penalty for revisiting cells
        
        # 5. Exploration bonus - small reward for visiting new cells
        else:
            reward += 0.5  # Small bonus for exploring new areas
        
        # 6. Energy efficiency - prefer downhill or flat terrain
        if elevation_change <= 0:  # Downhill or flat
            reward += abs(elevation_change) * 0.01  # Small bonus
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state."""
        row, col = self.current_pos
        current_elevation = self.elevation_map[row, col]
        
        # Get 5x5 elevation patch around current position
        patch = self._get_elevation_patch(row, col, patch_size=5)
        
        # Normalize coordinates
        norm_row = row / self.map_height
        norm_col = col / self.map_width
        
        # Combine observation components
        obs = np.concatenate([
            [norm_row, norm_col],           # Normalized position
            [current_elevation],            # Current elevation
            patch.flatten()                 # Flattened 5x5 elevation patch
        ]).astype(np.float32)
        
        return obs
    
    def _get_elevation_patch(self, center_row: int, center_col: int, patch_size: int = 5) -> np.ndarray:
        """Extract elevation patch around given position."""
        half_size = patch_size // 2
        patch = np.full((patch_size, patch_size), 0.0, dtype=np.float32)
        
        for i in range(patch_size):
            for j in range(patch_size):
                map_row = center_row - half_size + i
                map_col = center_col - half_size + j
                
                if (0 <= map_row < self.map_height and 0 <= map_col < self.map_width):
                    elevation = self.elevation_map[map_row, map_col]
                    if not np.isnan(elevation):
                        patch[i, j] = elevation
                    # else: keep as 0.0 for NaN values
                # else: keep as 0.0 for out-of-bounds
        
        return patch
    
    def _distance_to_goal(self) -> float:
        """Calculate Euclidean distance to goal."""
        return np.sqrt((self.current_pos[0] - self.goal_coord[0])**2 + 
                      (self.current_pos[1] - self.goal_coord[1])**2)
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == 'human':
            plt.figure(figsize=(10, 8))
            
            # Show elevation map
            plt.imshow(self.elevation_map, cmap='terrain', origin='upper')
            plt.colorbar(label='Elevation (m)')
            
            # Mark current position
            plt.plot(self.current_pos[1], self.current_pos[0], 'ro', markersize=10, label='Agent')
            
            # Mark goal
            plt.plot(self.goal_coord[1], self.goal_coord[0], 'g*', markersize=15, label='Goal')
            
            # Mark start if different from current
            if self.start_coord and self.current_pos != self.start_coord:
                plt.plot(self.start_coord[1], self.start_coord[0], 'bs', markersize=8, label='Start')
            
            plt.title(f'Terrain Navigation - Step: {self.step_count}')
            plt.xlabel('Column')
            plt.ylabel('Row')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
        elif mode == 'rgb_array':
            # Return rendered image as array for video recording (this all we do later)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(self.elevation_map, cmap='terrain', origin='upper')
            ax.plot(self.current_pos[1], self.current_pos[0], 'ro', markersize=8)
            ax.plot(self.goal_coord[1], self.goal_coord[0], 'g*', markersize=12)
            ax.set_title(f'Step: {self.step_count}')
            
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            
            return buf
    
    def close(self):
        """Clean up resources. ill do it later"""
        pass
