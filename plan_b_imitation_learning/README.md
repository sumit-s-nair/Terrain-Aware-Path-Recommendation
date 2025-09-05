# Plan B: Imitation Learning Approach

This folder contains a complete imitation learning pipeline as an alternative to pure reinforcement learning for training the pathfinding agent.

## 🎯 Approach

Instead of pure reinforcement learning, this method uses a three-step process:

1. **Expert Demonstration Extraction** (`trail_expert.py`)
   - Extracts optimal navigation patterns from actual trail coordinate data
   - Converts GPX trail data into state-action pairs showing expert behavior
   - Creates segmented demonstrations for manageable learning

2. **Behavioral Cloning Training** (`behavioral_cloning.py`) 
   - Trains neural network policy to mimic expert trail navigation
   - Uses supervised learning to predict actions from states
   - Includes proper train/validation splits and early stopping

3. **RL Fine-tuning** (`rl_finetune.py`)
   - Initializes PPO with behavioral cloning weights
   - Fine-tunes policy through environment interaction
   - Combines imitation learning with reinforcement learning benefits

## ✅ Advantages

- **Faster Convergence**: Learning from demonstrations accelerates training
- **Better Initialization**: Starts with reasonable navigation behavior instead of random
- **Sample Efficiency**: Requires fewer environment interactions than pure RL
- **Interpretable**: Can analyze what expert behavior the agent learned
- **Robust**: Less likely to get stuck in poor local optima

## 📁 Files

### Core Pipeline
- `trail_expert.py` - Extract expert demonstrations from trail coordinate data
- `behavioral_cloning.py` - Train neural network to imitate expert navigation
- `rl_finetune.py` - Fine-tune BC policy with PPO for optimization
- `run_imitation_pipeline.py` - Complete automated training pipeline

### Support Files
- `requirements.txt` - Additional Python package requirements
- `README.md` - This documentation

## 🚀 Quick Start

```bash
# Navigate to plan B directory
cd plan_b_imitation_learning

# Install additional requirements (if needed)
pip install torch scikit-learn matplotlib

# Run complete pipeline
python run_imitation_pipeline.py
```

## 🔧 Manual Step-by-Step

If you prefer to run steps individually:

```bash
# Step 1: Extract expert demonstrations
python trail_expert.py

# Step 2: Train behavioral cloning policy
python behavioral_cloning.py

# Step 3: Fine-tune with reinforcement learning
python rl_finetune.py
```

## 📊 Expected Outputs

The pipeline generates several important files:

- `trail_demonstrations.pkl` - Expert state-action pairs from trail data
- `best_navigation_policy.pth` - Behavioral cloning model (PyTorch)
- `best_finetuned_policy.zip` - Final RL fine-tuned model (Stable-Baselines3)
- `training_history.png` - Training curves showing learning progress

## 🧪 Testing

Use the fine-tuned model with the main testing script:

```bash
# Test the imitation learning model
cd ..  # Back to main directory
python test_agent.py --model plan_b_imitation_learning/best_finetuned_policy.zip
```

## 🔍 Comparison with Main Approach

This imitation learning approach should be compared with the main sophisticated RL training:

- **Main RL**: Pure reinforcement learning with enhanced reward structure
- **Plan B**: Imitation learning + RL fine-tuning for more stable convergence

Both approaches target the same goal: sophisticated long-distance pathfinding beyond simple single-action policies.
- [ ] Long-distance testing
