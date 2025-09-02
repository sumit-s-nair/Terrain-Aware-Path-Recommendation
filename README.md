# Terrain-Aware Path Recommendation

A physics-based reinforcement learning project for realistic hiking simulation using real elevation data. Features a custom environment that models realistic hiking physics including slip mechanics, energy systems, and terrain-aware movement costs.

---

## Features

- **Realistic Physics Environment**: Custom hiking simulation with slip mechanics, energy systems, and physics-based movement
- **Real Terrain Data**: Download and process high-resolution elevation data from USGS/OpenTopography
- **Fixed Reward Structure**: Progress-based rewards that prevent agent policy collapse and stuck behavior
- **Comprehensive Testing**: Behavior analysis and policy collapse detection for trained agents
- **Curriculum Learning**: Progressive training starting from easy distances with automatic progression
- **Complete Workflow**: Four-step pipeline from data download to agent testing

---

## Quick Start

1. **Clone and Install:**
   ```bash
   git clone <repository-url>
   cd Terrain-Aware-Path-Recommendation
   pip install -r requirements.txt
   ```

2. **Run Complete Workflow:**
   ```bash
   python 1.download_data.py      # Download terrain data
   python 2.preprocess_data.py    # Process for RL environment  
   python 3.build_and_train.py    # Train PPO agent with fixed rewards
   python 4.test_agent.py         # Test and analyze agent performance
   ```

---

## Core Workflow

### 1. Download Terrain Data (`1.download_data.py`)
Downloads high-resolution DEM and landcover data:
- Reads GPX trail file for area bounds
- Downloads 1-meter resolution DEM from USGS 3DEP
- Downloads landcover classification from NLCD
- Automatically tiles large areas for best resolution

### 2. Preprocess Data (`2.preprocess_data.py`)  
Converts raw data into RL-ready format:
- **slope.tif**: Terrain slope in degrees
- **stability.tif**: Ground stability (0-1 scale)
- **vegetation_cost.tif**: Movement cost based on landcover
- **terrain_rgb.tif**: RGB visualization for agent perception
- **trail_coordinates.npy**: Reference trail points

### 3. Train Agent (`3.build_and_train.py`)
Trains PPO agent with fixed reward structure:
- **Progress-based rewards**: 10x bonus for movement toward goal, 5x penalty for movement away
- **Anti-resting penalties**: Discourages agents from getting stuck
- **Curriculum learning**: Starts at 50m distance, automatically progresses
- **Physics simulation**: Realistic slip mechanics and energy systems

### 4. Test and Analyze (`4.test_agent.py`)
Comprehensive agent evaluation:
- Tests across multiple starting distances (50m, 100m, 200m, 500m)
- **Behavior analysis**: Detects policy collapse and stuck behavior
- **Action diversity tracking**: Ensures agents use varied strategies
- **Performance visualization**: Success rates, progress metrics, action patterns

---

## Key Improvements

### Fixed Reward Structure
The project implements a **progress-based reward system** that solves common RL problems:
- ✅ **Prevents Policy Collapse**: Agents learn diverse movement strategies
- ✅ **Eliminates Stuck Behavior**: Anti-resting penalties keep agents moving
- ✅ **Goal-Directed Learning**: Rewards progress toward goal, not just proximity
- ✅ **Stable Training**: Curriculum learning with automatic progression

### Realistic Physics
The `RealisticHikingEnv` includes:
- **Slip Mechanics**: Agents can slip on steep terrain based on slope and conditions
- **Energy System**: Movement costs vary by terrain difficulty
- **8-Direction Movement**: More realistic movement options than grid-based approaches
- **Health/Safety**: Agents must manage risk vs. progress

---

## File Structure

```
├── 1.download_data.py             # Download DEM and landcover data
├── 2.preprocess_data.py           # Process terrain for RL environment
├── 3.build_and_train.py           # Train PPO agent with fixed rewards
├── 4.test_agent.py                # Test and analyze agent performance
├── physics_hiking_env.py          # Core environment implementation
├── requirements.txt               # Python dependencies
├── data/
│   ├── raw/                       # DEM, landcover, GPX files
│   └── processed/                 # Processed terrain maps
├── outputs/
│   ├── checkpoints/               # Trained model files
│   └── tensorboard/               # Training logs
└── logs_fixed/                    # Training monitoring data
```

---

## Performance Analysis

The testing script provides detailed analysis:
- **Success Rate**: Percentage of episodes reaching the goal
- **Progress Metrics**: How far agents move toward the goal
- **Action Diversity**: Variety of movement strategies used
- **Policy Collapse Detection**: Identifies when agents only use one action
- **Behavioral Diagnosis**: Categorizes agent performance (excellent/good/poor/stuck)

---

## Troubleshooting

- **Policy Collapse**: If agents only use one action, retrain with the fixed reward structure
- **Stuck Behavior**: The progress-based rewards should prevent this; check reward implementation
- **No Models Found**: Run `3.build_and_train.py` to create trained models
- **Data Missing**: Ensure `1.download_data.py` and `2.preprocess_data.py` complete successfully

---

## Research Context

This project addresses key challenges in terrain-aware path planning:
1. **Realistic Physics**: Beyond simple grid-world navigation
2. **Real-World Data**: Using actual elevation and landcover data
3. **Stable RL Training**: Preventing common policy collapse issues
4. **Comprehensive Evaluation**: Detecting and diagnosing agent behaviors

The fixed reward structure is particularly important for preventing the common RL problem where agents learn to "game" the reward system by resting instead of making progress.
