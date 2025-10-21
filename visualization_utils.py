"""
Visualization Utilities for Training Results
Create plots for milestone report and final presentation

Author: Kip Chemweno
Course: ECE 4424 - Machine Learning
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd


class TrainingVisualizer:
    """Visualize training results and create report figures."""
    
    def __init__(self, history_path: str):
        """
        Initialize visualizer with training history.
        
        Args:
            history_path: Path to training_history.json file
        """
        with open(history_path, 'r') as f:
            self.history = json.load(f)
        
        self.save_dir = Path(history_path).parent / 'visualizations'
        self.save_dir.mkdir(exist_ok=True)
    
    def plot_training_curves(self, save_path: str = None):
        """Plot training and validation loss/MAE curves."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss curves
        axes[0].plot(epochs, self.history['train_loss'], 
                    'b-', linewidth=2, label='Training Loss')
        axes[0].plot(epochs, self.history['val_loss'], 
                    'r-', linewidth=2, label='Validation Loss')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('MSE Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # MAE curves
        axes[1].plot(epochs, self.history['train_mae'], 
                    'b-', linewidth=2, label='Training MAE')
        axes[1].plot(epochs, self.history['val_mae'], 
                    'r-', linewidth=2, label='Validation MAE')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Mean Absolute Error', fontsize=12)
        axes[1].set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / 'training_curves.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training curves to {save_path}")
        plt.show()
    
    def plot_learning_rate(self, save_path: str = None):
        """Plot learning rate schedule."""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        epochs = range(1, len(self.history['learning_rates']) + 1)
        
        ax.plot(epochs, self.history['learning_rates'], 
               'g-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / 'learning_rate.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved learning rate plot to {save_path}")
        plt.show()
    
    def create_milestone_report_figures(self):
        """Generate all figures needed for milestone report."""
        print("\nGenerating milestone report figures...")
        print("="*60)
        
        self.plot_training_curves(
            save_path=self.save_dir / 'milestone_training_curves.png'
        )
        
        self.plot_learning_rate(
            save_path=self.save_dir / 'milestone_lr_schedule.png'
        )
        
        # Summary statistics
        print("\nTraining Summary:")
        print(f"  Total epochs: {len(self.history['train_loss'])}")
        print(f"  Best validation loss: {min(self.history['val_loss']):.4f}")
        print(f"  Best validation MAE: {min(self.history['val_mae']):.4f}")
        print(f"  Final learning rate: {self.history['learning_rates'][-1]:.6f}")
        
        print(f"\n✓ All figures saved to {self.save_dir}")
        print("="*60)


def compare_three_models(real_history: str, 
                        carla_history: str, 
                        hybrid_history: str,
                        save_dir: str = './results/comparison'):
    """
    Compare training results from all three models.
    
    Args:
        real_history: Path to real-world model training history
        carla_history: Path to CARLA model training history
        hybrid_history: Path to hybrid model training history
        save_dir: Directory to save comparison plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load histories
    with open(real_history, 'r') as f:
        real_hist = json.load(f)
    with open(carla_history, 'r') as f:
        carla_hist = json.load(f)
    with open(hybrid_history, 'r') as f:
        hybrid_hist = json.load(f)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Validation Loss
    axes[0, 0].plot(real_hist['val_loss'], 'b-', linewidth=2, label='Real-world')
    axes[0, 0].plot(carla_hist['val_loss'], 'r-', linewidth=2, label='CARLA')
    axes[0, 0].plot(hybrid_hist['val_loss'], 'g-', linewidth=2, label='Hybrid')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Validation Loss', fontsize=12)
    axes[0, 0].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation MAE
    axes[0, 1].plot(real_hist['val_mae'], 'b-', linewidth=2, label='Real-world')
    axes[0, 1].plot(carla_hist['val_mae'], 'r-', linewidth=2, label='CARLA')
    axes[0, 1].plot(hybrid_hist['val_mae'], 'g-', linewidth=2, label='Hybrid')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Validation MAE', fontsize=12)
    axes[0, 1].set_title('Validation MAE Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training Loss
    axes[1, 0].plot(real_hist['train_loss'], 'b-', linewidth=2, label='Real-world')
    axes[1, 0].plot(carla_hist['train_loss'], 'r-', linewidth=2, label='CARLA')
    axes[1, 0].plot(hybrid_hist['train_loss'], 'g-', linewidth=2, label='Hybrid')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Training Loss', fontsize=12)
    axes[1, 0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bar chart of best metrics
    models = ['Real-world', 'CARLA', 'Hybrid']
    best_val_loss = [min(real_hist['val_loss']), 
                     min(carla_hist['val_loss']), 
                     min(hybrid_hist['val_loss'])]
    
    axes[1, 1].bar(models, best_val_loss, color=['blue', 'red', 'green'], 
                   alpha=0.7, edgecolor='black', linewidth=2)
    axes[1, 1].set_ylabel('Best Validation Loss', fontsize=12)
    axes[1, 1].set_title('Best Model Performance', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(best_val_loss):
        axes[1, 1].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'three_model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison to {save_dir / 'three_model_comparison.png'}")
    plt.show()
    
    # Print summary table
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    comparison_df = pd.DataFrame({
        'Model': models,
        'Best Val Loss': best_val_loss,
        'Best Val MAE': [min(real_hist['val_mae']), 
                        min(carla_hist['val_mae']), 
                        min(hybrid_hist['val_mae'])],
        'Final Train Loss': [real_hist['train_loss'][-1],
                            carla_hist['train_loss'][-1],
                            hybrid_hist['train_loss'][-1]],
        'Epochs': [len(real_hist['train_loss']),
                  len(carla_hist['train_loss']),
                  len(hybrid_hist['train_loss'])]
    })
    
    print(comparison_df.to_string(index=False))
    print("="*80 + "\n")
    
    # Save to CSV
    comparison_df.to_csv(save_dir / 'model_comparison_summary.csv', index=False)
    print(f"✓ Saved summary to {save_dir / 'model_comparison_summary.csv'}")


if __name__ == "__main__":
    print("Visualization utilities module loaded")
    print("\nExample usage:")
    print("  vis = TrainingVisualizer('./checkpoints/experiment/training_history.json')")
    print("  vis.create_milestone_report_figures()")
