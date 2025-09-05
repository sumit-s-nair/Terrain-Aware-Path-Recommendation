# trail_expert.py
"""
Extract expert demonstrations from trail data for imitation learning.
Converts GPX/trail coordinate data into state-action pairs that show optimal navigation.
"""

import numpy as np
import rasterio
import rasterio.transform
from pathlib import Path
from typing import List, Tuple, Dict
import pickle

class TrailExpert:
    """Extracts expert demonstrations from trail data"""
    
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = Path(data_dir)
        self.trail_coords = None
        self.cell_size = 2.0  # meters per pixel
        
        # Load trail coordinates
        trail_path = self.data_dir / "processed" / "trail_coordinates.npy"
        if trail_path.exists():
            self.trail_coords = np.load(trail_path)
            print(f"✅ Loaded {len(self.trail_coords)} trail points")
        else:
            raise FileNotFoundError(f"Trail coordinates not found at {trail_path}")
    
    def extract_demonstrations(self, segment_length: int = 100) -> List[Dict]:
        """
        Extract expert demonstrations from trail data.
        
        Args:
            segment_length: Number of trail points per demonstration segment
            
        Returns:
            List of demonstrations, each containing states and actions
        """
        demonstrations = []
        
        # Split trail into segments for manageable demonstrations
        num_segments = len(self.trail_coords) // segment_length
        
        for segment_idx in range(num_segments):
            start_idx = segment_idx * segment_length
            end_idx = min(start_idx + segment_length, len(self.trail_coords))
            segment = self.trail_coords[start_idx:end_idx]
            
            if len(segment) < 10:  # Skip very short segments
                continue
            
            states = []
            actions = []
            
            # Extract state-action pairs from trail segment
            for i in range(len(segment) - 1):
                current_pos = segment[i]
                next_pos = segment[i + 1]
                
                # Create state (position relative to goal)
                goal_pos = segment[-1]  # End of segment as goal
                state = self._create_state(current_pos, goal_pos, segment)
                
                # Determine action (movement direction)
                action = self._pos_to_action(current_pos, next_pos)
                
                if action is not None:  # Valid action
                    states.append(state)
                    actions.append(action)
            
            if len(states) > 0:
                demonstration = {
                    'states': np.array(states),
                    'actions': np.array(actions),
                    'segment_idx': segment_idx,
                    'trail_section': segment
                }
                demonstrations.append(demonstration)
        
        print(f"✅ Extracted {len(demonstrations)} demonstrations")
        return demonstrations
    
    def _create_state(self, current_pos: np.ndarray, goal_pos: np.ndarray, trail_segment: np.ndarray) -> np.ndarray:
        """Create state representation for imitation learning"""
        
        # Basic state: position and goal direction
        goal_direction = goal_pos - current_pos
        goal_distance = np.linalg.norm(goal_direction)
        
        if goal_distance > 0:
            goal_direction_norm = goal_direction / goal_distance
        else:
            goal_direction_norm = np.array([0.0, 0.0])
        
        # Add trail context - direction to next few trail points
        trail_context = np.zeros(4)  # Next 2 trail points as context
        current_idx = self._find_closest_trail_point(current_pos, trail_segment)
        
        for i, offset in enumerate([1, 2]):
            next_idx = current_idx + offset
            if next_idx < len(trail_segment):
                direction = trail_segment[next_idx] - current_pos
                distance = np.linalg.norm(direction)
                if distance > 0:
                    trail_context[i*2:(i+1)*2] = direction / distance
        
        # Combine into state vector
        state = np.concatenate([
            current_pos / 1000.0,  # Normalized position
            goal_direction_norm,    # Direction to goal
            [goal_distance / 1000.0],  # Distance to goal
            trail_context           # Trail context
        ])
        
        return state.astype(np.float32)
    
    def _find_closest_trail_point(self, pos: np.ndarray, trail_segment: np.ndarray) -> int:
        """Find closest trail point index"""
        distances = np.linalg.norm(trail_segment - pos, axis=1)
        return np.argmin(distances)
    
    def _pos_to_action(self, current_pos: np.ndarray, next_pos: np.ndarray) -> int:
        """Convert position movement to discrete action"""
        
        movement = next_pos - current_pos
        distance = np.linalg.norm(movement)
        
        if distance < 0.1:  # Too small movement
            return None
        
        # Normalize movement
        direction = movement / distance
        
        # Convert to 8-directional action
        # Actions: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
        angle = np.arctan2(direction[1], direction[0])  # col, row
        angle_deg = np.degrees(angle) % 360
        
        # Convert angle to 8-direction action
        # Adjust for coordinate system where -row is up
        angle_deg = (angle_deg + 180) % 360  # Adjust for row coordinate
        
        action_angles = [0, 45, 90, 135, 180, 225, 270, 315]  # N, NE, E, SE, S, SW, W, NW
        
        # Find closest action
        angle_diffs = [abs(angle_deg - a) for a in action_angles]
        # Handle wrap-around
        angle_diffs = [min(diff, 360 - diff) for diff in angle_diffs]
        
        action = np.argmin(angle_diffs)
        return action
    
    def save_demonstrations(self, demonstrations: List[Dict], filename: str = "trail_demonstrations.pkl"):
        """Save demonstrations to file"""
        filepath = Path(filename)
        with open(filepath, 'wb') as f:
            pickle.dump(demonstrations, f)
        print(f"💾 Saved {len(demonstrations)} demonstrations to {filepath}")
    
    def load_demonstrations(self, filename: str = "trail_demonstrations.pkl") -> List[Dict]:
        """Load demonstrations from file"""
        filepath = Path(filename)
        with open(filepath, 'rb') as f:
            demonstrations = pickle.load(f)
        print(f"📂 Loaded {len(demonstrations)} demonstrations from {filepath}")
        return demonstrations

def main():
    """Extract and save expert demonstrations"""
    
    print("🎯 EXTRACTING TRAIL EXPERT DEMONSTRATIONS")
    print("=" * 50)
    
    # Create expert and extract demonstrations
    expert = TrailExpert()
    demonstrations = expert.extract_demonstrations(segment_length=50)
    
    # Analyze demonstrations
    total_examples = sum(len(demo['states']) for demo in demonstrations)
    print(f"📊 Total state-action pairs: {total_examples}")
    
    if demonstrations:
        # Show action distribution
        all_actions = np.concatenate([demo['actions'] for demo in demonstrations])
        action_counts = np.bincount(all_actions, minlength=8)
        action_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        
        print("\n📈 Action Distribution:")
        for i, (name, count) in enumerate(zip(action_names, action_counts)):
            percent = count / len(all_actions) * 100
            print(f"  Action {i} ({name}): {count:4d} ({percent:5.1f}%)")
    
    # Save demonstrations
    expert.save_demonstrations(demonstrations)
    
    print("\n✅ Expert demonstrations ready for imitation learning!")

if __name__ == "__main__":
    main()
