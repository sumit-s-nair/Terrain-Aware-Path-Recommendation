# behavioral_cloning.py
"""
Behavioral Cloning for imitation learning.
Train agent to mimic expert trail navigation behavior.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class TrailDataset(Dataset):
    """Dataset for trail navigation demonstrations"""
    
    def __init__(self, demonstrations: List[Dict]):
        self.states = []
        self.actions = []
        
        # Combine all demonstrations
        for demo in demonstrations:
            self.states.extend(demo['states'])
            self.actions.extend(demo['actions'])
        
        self.states = np.array(self.states, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.int64)
        
        print(f"📊 Dataset: {len(self.states)} examples, state dim: {self.states.shape[1]}")
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return torch.tensor(self.states[idx]), torch.tensor(self.actions[idx])

class NavigationPolicy(nn.Module):
    """Neural network policy for trail navigation"""
    
    def __init__(self, state_dim: int, action_dim: int = 8, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, state):
        return self.network(state)
    
    def predict_action(self, state):
        """Predict action for given state"""
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.tensor(state, dtype=torch.float32)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            
            logits = self.forward(state)
            action = torch.argmax(logits, dim=1)
            return action.numpy() if action.numel() == 1 else action.numpy()

class BehavioralCloningTrainer:
    """Trainer for behavioral cloning"""
    
    def __init__(self, model: NavigationPolicy, learning_rate: float = 0.001, device: str = None):
        self.model = model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        print(f"🚀 Training on {self.device}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for states, actions in train_loader:
            states = states.to(self.device)
            actions = actions.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(states)
            loss = self.criterion(logits, actions)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == actions).sum().item()
            total_predictions += actions.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for states, actions in val_loader:
                states = states.to(self.device)
                actions = actions.to(self.device)
                
                logits = self.model(states)
                loss = self.criterion(logits, actions)
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == actions).sum().item()
                total_predictions += actions.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 100):
        """Full training loop"""
        
        print(f"🎯 Training for {epochs} epochs...")
        best_val_accuracy = 0
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train and validate
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0
                self.save_model("best_navigation_policy.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"🛑 Early stopping at epoch {epoch}")
                    break
        
        print(f"✅ Training complete! Best validation accuracy: {best_val_accuracy:.4f}")
        return best_val_accuracy
    
    def save_model(self, filepath: str):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        print(f"📂 Loaded model from {filepath}")
    
    def plot_training_history(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Training Loss', alpha=0.7)
        ax1.plot(self.val_losses, label='Validation Loss', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Training Accuracy', alpha=0.7)
        ax2.plot(self.val_accuracies, label='Validation Accuracy', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Train behavioral cloning policy"""
    
    print("🎯 BEHAVIORAL CLONING TRAINING")
    print("=" * 50)
    
    # Load demonstrations
    from trail_expert import TrailExpert
    
    try:
        expert = TrailExpert()
        demonstrations = expert.load_demonstrations()
    except FileNotFoundError:
        print("📥 Extracting demonstrations first...")
        expert = TrailExpert()
        demonstrations = expert.extract_demonstrations()
        expert.save_demonstrations(demonstrations)
    
    # Create dataset
    dataset = TrailDataset(demonstrations)
    state_dim = dataset.states.shape[1]
    
    # Split data
    train_states, val_states, train_actions, val_actions = train_test_split(
        dataset.states, dataset.actions, test_size=0.2, random_state=42, stratify=dataset.actions
    )
    
    # Create data loaders
    train_dataset = TrailDataset([{
        'states': train_states,
        'actions': train_actions
    }])
    val_dataset = TrailDataset([{
        'states': val_states, 
        'actions': val_actions
    }])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create model and trainer
    model = NavigationPolicy(state_dim=state_dim)
    trainer = BehavioralCloningTrainer(model)
    
    # Train model
    best_accuracy = trainer.train(train_loader, val_loader, epochs=200)
    
    # Plot results
    trainer.plot_training_history()
    
    # Test on validation set
    print("\n📊 FINAL EVALUATION:")
    val_predictions = []
    val_true = []
    
    model.eval()
    with torch.no_grad():
        for states, actions in val_loader:
            states = states.to(trainer.device)
            logits = model(states)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            val_predictions.extend(predictions)
            val_true.extend(actions.numpy())
    
    # Classification report
    action_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    print(f"\nFinal Accuracy: {accuracy_score(val_true, val_predictions):.4f}")
    print("\nPer-Action Performance:")
    print(classification_report(val_true, val_predictions, target_names=action_names))
    
    print(f"\n✅ Behavioral cloning complete! Best accuracy: {best_accuracy:.4f}")
    print("💾 Model saved as 'best_navigation_policy.pth'")

if __name__ == "__main__":
    main()
