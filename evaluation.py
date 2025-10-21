"""
Evaluation and Testing Framework for Lane Keeping Models
Implements metrics: MAE, RMSE, and qualitative assessment

Author: Kip Chemweno
Course: ECE 4424 - Machine Learning
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple
import json
from pathlib import Path


class ModelEvaluator:
    """Comprehensive model evaluation for steering prediction."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def evaluate_on_loader(self, data_loader) -> Dict[str, float]:
        """
        Evaluate model on a data loader.
        
        Returns:
            Dictionary with evaluation metrics
        """
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, steering_angles in data_loader:
                images = images.to(self.device)
                steering_angles = steering_angles.to(self.device).unsqueeze(1)
                
                # Predict
                predictions = self.model(images)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(steering_angles.cpu().numpy())
        
        predictions = np.array(all_predictions).flatten()
        targets = np.array(all_targets).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)
        
        # Mean absolute percentage error (careful with division by zero)
        non_zero_mask = targets != 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((targets[non_zero_mask] - predictions[non_zero_mask]) 
                                  / targets[non_zero_mask])) * 100
        else:
            mape = float('inf')
        
        # Accuracy within threshold
        threshold_5 = np.mean(np.abs(predictions - targets) < 0.05) * 100
        threshold_10 = np.mean(np.abs(predictions - targets) < 0.10) * 100
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'mape': float(mape),
            'accuracy_5deg': float(threshold_5),
            'accuracy_10deg': float(threshold_10),
            'n_samples': len(predictions)
        }
        
        return metrics, predictions, targets
    
    def print_metrics(self, metrics: Dict[str, float], dataset_name: str = "Test"):
        """Print evaluation metrics in a formatted way."""
        print(f"\n{'='*60}")
        print(f"{dataset_name.upper()} SET EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Samples evaluated: {metrics['n_samples']:,}")
        print(f"\n--- Regression Metrics ---")
        print(f"MAE (Mean Absolute Error):     {metrics['mae']:.4f}")
        print(f"RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}")
        print(f"R² Score:                      {metrics['r2_score']:.4f}")
        if metrics['mape'] != float('inf'):
            print(f"MAPE (Mean Abs % Error):       {metrics['mape']:.2f}%")
        
        print(f"\n--- Accuracy Thresholds ---")
        print(f"Within ±0.05 radians (~3°):    {metrics['accuracy_5deg']:.2f}%")
        print(f"Within ±0.10 radians (~6°):    {metrics['accuracy_10deg']:.2f}%")
        print(f"{'='*60}\n")
    
    def plot_predictions(self, predictions: np.ndarray, targets: np.ndarray, 
                        save_path: str = None, dataset_name: str = "Test"):
        """Create visualization of predictions vs targets."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Scatter plot: Predictions vs Targets
        axes[0, 0].scatter(targets, predictions, alpha=0.5, s=10)
        
        # Add perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 
                       'r--', linewidth=2, label='Perfect Prediction')
        
        axes[0, 0].set_xlabel('True Steering Angle')
        axes[0, 0].set_ylabel('Predicted Steering Angle')
        axes[0, 0].set_title(f'Predictions vs Ground Truth ({dataset_name})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Error distribution
        errors = predictions - targets
        axes[0, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].axvline(x=errors.mean(), color='g', linestyle='--', 
                          linewidth=2, label=f'Mean: {errors.mean():.4f}')
        axes[0, 1].set_xlabel('Prediction Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Error Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Absolute error distribution
        abs_errors = np.abs(errors)
        axes[1, 0].hist(abs_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 0].axvline(x=abs_errors.mean(), color='r', linestyle='--', 
                          linewidth=2, label=f'MAE: {abs_errors.mean():.4f}')
        axes[1, 0].set_xlabel('Absolute Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Absolute Error Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Residual plot
        axes[1, 1].scatter(predictions, errors, alpha=0.5, s=10)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Predicted Steering Angle')
        axes[1, 1].set_ylabel('Residual (Error)')
        axes[1, 1].set_title('Residual Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved predictions plot to {save_path}")
        
        plt.show()
    
    def analyze_error_by_steering_range(self, predictions: np.ndarray, 
                                       targets: np.ndarray) -> Dict:
        """Analyze error patterns across different steering ranges."""
        # Define steering ranges
        ranges = {
            'Straight (-0.05 to 0.05)': (-0.05, 0.05),
            'Slight Left (-0.15 to -0.05)': (-0.15, -0.05),
            'Slight Right (0.05 to 0.15)': (0.05, 0.15),
            'Sharp Left (< -0.15)': (float('-inf'), -0.15),
            'Sharp Right (> 0.15)': (0.15, float('inf'))
        }
        
        print(f"\n{'='*60}")
        print("ERROR ANALYSIS BY STEERING RANGE")
        print(f"{'='*60}")
        
        results = {}
        
        for range_name, (lower, upper) in ranges.items():
            mask = (targets >= lower) & (targets < upper)
            n_samples = mask.sum()
            
            if n_samples > 0:
                range_targets = targets[mask]
                range_predictions = predictions[mask]
                range_errors = range_predictions - range_targets
                
                mae = np.abs(range_errors).mean()
                rmse = np.sqrt((range_errors ** 2).mean())
                
                results[range_name] = {
                    'n_samples': int(n_samples),
                    'percentage': float(n_samples / len(targets) * 100),
                    'mae': float(mae),
                    'rmse': float(rmse)
                }
                
                print(f"\n{range_name}")
                print(f"  Samples: {n_samples} ({n_samples/len(targets)*100:.1f}%)")
                print(f"  MAE: {mae:.4f}")
                print(f"  RMSE: {rmse:.4f}")
            else:
                print(f"\n{range_name}: No samples")
        
        print(f"{'='*60}\n")
        return results


def compare_models(models_dict: Dict[str, nn.Module], 
                   test_loader,
                   device: str = 'cuda',
                   save_path: str = './results/model_comparison.png'):
    """
    Compare multiple models side by side.
    
    Args:
        models_dict: Dictionary of model_name -> model
        test_loader: Test data loader
        device: Device to run on
        save_path: Where to save comparison plot
    """
    results = {}
    
    print(f"\n{'='*80}")
    print("COMPARING MODELS")
    print(f"{'='*80}\n")
    
    for model_name, model in models_dict.items():
        print(f"Evaluating {model_name}...")
        evaluator = ModelEvaluator(model, device)
        metrics, predictions, targets = evaluator.evaluate_on_loader(test_loader)
        evaluator.print_metrics(metrics, model_name)
        
        results[model_name] = {
            'metrics': metrics,
            'predictions': predictions,
            'targets': targets
        }
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    model_names = list(results.keys())
    metrics_to_plot = ['mae', 'rmse', 'r2_score']
    titles = ['Mean Absolute Error', 'Root Mean Squared Error', 'R² Score']
    
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        values = [results[name]['metrics'][metric] for name in model_names]
        
        axes[idx].bar(model_names, values, alpha=0.7, edgecolor='black')
        axes[idx].set_ylabel(metric.upper())
        axes[idx].set_title(title)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to {save_path}")
    plt.show()
    
    # Save results to JSON
    json_path = Path(save_path).parent / 'model_comparison.json'
    json_results = {
        name: {'metrics': data['metrics']} 
        for name, data in results.items()
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=4)
    
    print(f"Saved comparison results to {json_path}")
    
    return results


if __name__ == "__main__":
    print("Evaluation module loaded successfully")
