# rl_finetune.py
"""
Fine-tune behavioral cloning policy with RL for improved performance.
Combines imitation learning with environment interaction.
"""

import numpy as np
import torch
import sys
from pathlib import Path
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from behavioral_cloning import NavigationPolicy, BehavioralCloningTrainer
from gymnasium import spaces
import gymnasium as gym

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from physics_hiking_env import RealisticHikingEnv

class BCInitializedPolicy(BaseFeaturesExtractor):
    """
    Custom feature extractor that initializes from behavioral cloning policy
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Calculate input dimension from observation space
        if isinstance(observation_space, spaces.Dict):
            total_dim = 0
            for key, space in observation_space.spaces.items():
                if isinstance(space, spaces.Box):
                    total_dim += np.prod(space.shape)
        else:
            total_dim = np.prod(observation_space.shape)
        
        self.bc_policy = None  # Will be loaded later
        self.features_dim = features_dim
        
        # Feature extraction network (will be initialized from BC policy)
        self.feature_extractor = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations) -> torch.Tensor:
        # Flatten observations if dict
        if isinstance(observations, dict):
            obs_list = []
            for key in sorted(observations.keys()):
                obs_tensor = observations[key]
                if len(obs_tensor.shape) > 2:
                    obs_tensor = obs_tensor.flatten(start_dim=1)
                obs_list.append(obs_tensor)
            flattened_obs = torch.cat(obs_list, dim=1)
        else:
            flattened_obs = observations.flatten(start_dim=1)
        
        return self.feature_extractor(flattened_obs)
    
    def load_bc_weights(self, bc_model_path: str):
        """Load weights from behavioral cloning model"""
        try:
            checkpoint = torch.load(bc_model_path, map_location='cpu')
            
            # Create a dummy BC model to extract weights
            dummy_bc = NavigationPolicy(state_dim=7)  # Adjust based on your state dim
            dummy_bc.load_state_dict(checkpoint['model_state_dict'])
            
            # Transfer compatible weights
            self._transfer_weights(dummy_bc)
            print(f"✅ Loaded BC weights from {bc_model_path}")
            
        except Exception as e:
            print(f"⚠️  Could not load BC weights: {e}")
            print("🔄 Proceeding with random initialization")
    
    def _transfer_weights(self, bc_model):
        """Transfer weights from BC model to feature extractor"""
        bc_layers = list(bc_model.network.children())
        fe_layers = list(self.feature_extractor.children())
        
        # Transfer compatible linear layers
        bc_linear_idx = 0
        for i, fe_layer in enumerate(fe_layers):
            if isinstance(fe_layer, nn.Linear):
                # Find next linear layer in BC model
                while bc_linear_idx < len(bc_layers) and not isinstance(bc_layers[bc_linear_idx], nn.Linear):
                    bc_linear_idx += 1
                
                if bc_linear_idx < len(bc_layers):
                    bc_linear = bc_layers[bc_linear_idx]
                    
                    # Transfer weights if dimensions match
                    if (bc_linear.weight.shape == fe_layer.weight.shape and 
                        bc_linear.bias.shape == fe_layer.bias.shape):
                        fe_layer.weight.data.copy_(bc_linear.weight.data)
                        fe_layer.bias.data.copy_(bc_linear.bias.data)
                        print(f"  Transferred layer {i} weights: {fe_layer.weight.shape}")
                    
                    bc_linear_idx += 1

class FineTuningCallback(BaseCallback):
    """Callback for monitoring RL fine-tuning progress"""
    
    def __init__(self, eval_env, eval_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.evaluations = []
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate policy
            mean_reward = self._evaluate_policy()
            self.evaluations.append(mean_reward)
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                # Save best model
                self.model.save("best_finetuned_policy")
                
            if self.verbose > 0:
                print(f"Step {self.n_calls}: Mean reward: {mean_reward:.2f}, Best: {self.best_mean_reward:.2f}")
        
        return True
    
    def _evaluate_policy(self, n_eval_episodes: int = 10):
        """Evaluate current policy"""
        episode_rewards = []
        
        for _ in range(n_eval_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = self.eval_env.step(action)
                episode_reward += reward
                done = done or truncated
            
            episode_rewards.append(episode_reward)
        
        return np.mean(episode_rewards)

class ImitationToRLTrainer:
    """Trainer for fine-tuning BC policy with RL"""
    
    def __init__(self, bc_model_path: str = "best_navigation_policy.pth"):
        self.bc_model_path = bc_model_path
        
        # Create training environment
        self.train_env = RealisticHikingEnv()
        print(f"🌲 Training environment created")
        
        # Create evaluation environment
        self.eval_env = RealisticHikingEnv()
        print(f"🔍 Evaluation environment created")
    
    def create_rl_model(self):
        """Create PPO model with BC-initialized policy"""
        
        # Policy kwargs for custom feature extractor
        policy_kwargs = {
            "features_extractor_class": BCInitializedPolicy,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": [256, 256]  # Additional layers after feature extractor
        }
        
        # Create PPO model
        model = PPO(
            "MultiInputPolicy",
            self.train_env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Moderate exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log="./tensorboard_finetune/"
        )
        
        # Load BC weights into feature extractor
        if Path(self.bc_model_path).exists():
            feature_extractor = model.policy.features_extractor
            feature_extractor.load_bc_weights(self.bc_model_path)
        else:
            print(f"⚠️  BC model not found at {self.bc_model_path}")
            print("🔄 Proceeding with random initialization")
        
        return model
    
    def finetune(self, timesteps: int = 500000):
        """Fine-tune BC policy with RL"""
        
        print("🎯 RL FINE-TUNING FROM BEHAVIORAL CLONING")
        print("=" * 60)
        
        # Create model
        model = self.create_rl_model()
        
        # Create callback for evaluation
        callback = FineTuningCallback(
            eval_env=self.eval_env,
            eval_freq=10000,
            verbose=1
        )
        
        print(f"🚀 Starting RL fine-tuning for {timesteps:,} timesteps")
        
        # Train model
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            progress_bar=True
        )
        
        # Save final model
        model.save("final_finetuned_policy")
        print(f"💾 Saved final model to 'final_finetuned_policy.zip'")
        
        # Final evaluation
        print("\n🎯 FINAL EVALUATION:")
        final_reward = callback._evaluate_policy(n_eval_episodes=20)
        print(f"Final mean reward: {final_reward:.2f}")
        print(f"Best mean reward during training: {callback.best_mean_reward:.2f}")
        
        return model, callback.evaluations
    
    def test_finetuned_policy(self, model_path: str = "best_finetuned_policy.zip", n_episodes: int = 10):
        """Test the fine-tuned policy"""
        
        print(f"\n🧪 Testing fine-tuned policy from {model_path}")
        
        # Load model
        model = PPO.load(model_path)
        
        episode_rewards = []
        success_count = 0
        
        for episode in range(n_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 10000:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                steps += 1
                done = done or truncated
            
            episode_rewards.append(episode_reward)
            if info.get('goal_reached', False):
                success_count += 1
            
            print(f"Episode {episode + 1}: Reward: {episode_reward:.2f}, Steps: {steps}, Success: {info.get('goal_reached', False)}")
        
        mean_reward = np.mean(episode_rewards)
        success_rate = success_count / n_episodes
        
        print(f"\n📊 RESULTS:")
        print(f"Mean reward: {mean_reward:.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Success rate: {success_rate:.2%} ({success_count}/{n_episodes})")
        
        return mean_reward, success_rate

def main():
    """Main training and evaluation pipeline"""
    
    print("🎯 IMITATION + RL FINE-TUNING PIPELINE")
    print("=" * 60)
    
    # Check if BC model exists
    bc_model_path = "best_navigation_policy.pth"
    if not Path(bc_model_path).exists():
        print(f"❌ BC model not found at {bc_model_path}")
        print("🔧 Please run behavioral_cloning.py first to train the base policy")
        
        # Offer to run BC training
        response = input("\n🤔 Would you like to run behavioral cloning first? (y/n): ")
        if response.lower() in ['y', 'yes']:
            print("🚀 Running behavioral cloning...")
            from behavioral_cloning import main as bc_main
            bc_main()
        else:
            print("⚠️  Proceeding without BC initialization")
    
    # Create trainer and fine-tune
    trainer = ImitationToRLTrainer(bc_model_path)
    model, evaluations = trainer.finetune(timesteps=500000)
    
    # Test the fine-tuned policy
    trainer.test_finetuned_policy()
    
    print("\n✅ RL fine-tuning complete!")
    print("💡 Use 'best_finetuned_policy.zip' for best performance")
    print("💡 Use 'final_finetuned_policy.zip' for final trained model")

if __name__ == "__main__":
    main()
