"""
Compare All Three Models: Real, CARLA, and Hybrid
Comprehensive evaluation and visualization for final report

Author: Kip Chemweno
Course: ECE 4424 - Machine Learning
Date: November 2025

Usage:
    python compare_all_models.py
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List
import argparse

from model import PilotNet
from evaluation import ModelEvaluator
from data_pipeline import create_dataloaders


class ThreeModelComparison:
    """Compare Real, CARLA, and Hybrid models comprehensively."""
    
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.results = {}
        self.output_dir = Path('./results/three_model_comparison')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model(self, checkpoint_path: str, model_name: str) -> PilotNet:
        """Load a trained model from checkpoint."""
        print(f"\nLoading {model_name} model from {checkpoint_path}...")
        
        if not Path(checkpoint_path).exists():
            print(f"✗ Checkpoint not found: {checkpoint_path}")
            return None
        
        model = PilotNet()
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"✓ Loaded {model_name} (epoch {checkpoint['epoch']})")
        return model
    
    def evaluate_model(self, model: PilotNet, data_loader, model_name: str) -> Dict:
        """Evaluate a model and return metrics."""
        print(f"\nEvaluating {model_name}...")
        
        evaluator = ModelEvaluator(model, self.device)
        metrics, predictions, targets = evaluator.evaluate_on_loader(data_loader)
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'predictions': predictions,
            'targets': targets
        }
        
        print(f"✓ {model_name} MAE: {metrics['mae']:.4f} rad ({metrics['mae']*57.3:.2f}°)")
        print(f"✓ {model_name} Accuracy (±3°): {metrics['accuracy_5deg']:.1f}%")
        
        return metrics
    
    def create_comparison_table(self):
        """Create comparison table of all models."""
        print(f"\n{'='*80}")
        print("MODEL COMPARISON TABLE")
        print(f"{'='*80}")
        
        # Create dataframe
        data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            data.append({
                'Model': model_name,
                'MAE (rad)': f"{metrics['mae']:.4f}",
                'MAE (°)': f"{metrics['mae']*57.3:.2f}",
                'RMSE': f"{metrics['rmse']:.4f}",
                'R² Score': f"{metrics['r2_score']:.4f}",
                'Acc ±3°': f"{metrics['accuracy_5deg']:.1f}%",
                'Acc ±6°': f"{metrics['accuracy_10deg']:.1f}%",
                'Samples': metrics['n_samples']
            })
        
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        print(f"{'='*80}\n")
        
        # Save to CSV
        csv_path = self.output_dir / 'model_comparison_table.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved comparison table: {csv_path}")
        
        return df
    
    def plot_metrics_comparison(self):
        """Plot bar charts comparing key metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        model_names = list(self.results.keys())
        colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green
        
        metrics_to_plot = [
            ('mae', 'Mean Absolute Error (rad)', axes[0, 0]),
            ('rmse', 'Root Mean Squared Error (rad)', axes[0, 1]),
            ('r2_score', 'R² Score', axes[0, 2]),
            ('accuracy_5deg', 'Accuracy ±3° (%)', axes[1, 0]),
            ('accuracy_10deg', 'Accuracy ±6° (%)', axes[1, 1]),
        ]
        
        for metric_key, title, ax in metrics_to_plot:
            values = [self.results[name]['metrics'][metric_key] for name in model_names]
            bars = ax.bar(model_names, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
            
            ax.set_ylabel(metric_key.upper().replace('_', ' '), fontsize=12)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}' if metric_key in ['mae', 'rmse', 'r2_score'] else f'{val:.1f}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Sixth subplot: Summary text
        axes[1, 2].axis('off')
        summary_text = "Model Rankings:\n\n"
        
        # Rank by MAE (lower is better)
        mae_ranking = sorted(self.results.items(), 
                           key=lambda x: x[1]['metrics']['mae'])
        summary_text += "By MAE (Lower Better):\n"
        for rank, (name, data) in enumerate(mae_ranking, 1):
            summary_text += f"  {rank}. {name}: {data['metrics']['mae']:.4f}\n"
        
        summary_text += "\nBy Accuracy ±3° (Higher Better):\n"
        # Rank by accuracy (higher is better)
        acc_ranking = sorted(self.results.items(), 
                           key=lambda x: x[1]['metrics']['accuracy_5deg'], 
                           reverse=True)
        for rank, (name, data) in enumerate(acc_ranking, 1):
            summary_text += f"  {rank}. {name}: {data['metrics']['accuracy_5deg']:.1f}%\n"
        
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, 
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Three-Model Performance Comparison', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        save_path = self.output_dir / 'metrics_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved metrics comparison: {save_path}")
        plt.close()
    
    def plot_prediction_scatter(self):
        """Create scatter plots for all three models."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        model_names = list(self.results.keys())
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for idx, (model_name, color) in enumerate(zip(model_names, colors)):
            predictions = self.results[model_name]['predictions']
            targets = self.results[model_name]['targets']
            
            # Scatter plot
            axes[idx].scatter(targets, predictions, alpha=0.5, s=10, color=color)
            
            # Perfect prediction line
            min_val = min(targets.min(), predictions.min())
            max_val = max(targets.max(), predictions.max())
            axes[idx].plot([min_val, max_val], [min_val, max_val], 
                          'r--', linewidth=2, label='Perfect Prediction')
            
            # Calculate R²
            r2 = self.results[model_name]['metrics']['r2_score']
            mae = self.results[model_name]['metrics']['mae']
            
            axes[idx].set_xlabel('True Steering Angle (rad)', fontsize=11)
            axes[idx].set_ylabel('Predicted Steering Angle (rad)', fontsize=11)
            axes[idx].set_title(f'{model_name}\nR²={r2:.4f}, MAE={mae:.4f}', 
                               fontsize=12, fontweight='bold')
            axes[idx].legend(fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_aspect('equal', adjustable='box')
        
        plt.suptitle('Prediction Scatter Plots - Three Models', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = self.output_dir / 'prediction_scatter_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved prediction scatter: {save_path}")
        plt.close()
    
    def plot_error_distributions(self):
        """Plot error distributions for all models."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        model_names = list(self.results.keys())
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for idx, (model_name, color) in enumerate(zip(model_names, colors)):
            predictions = self.results[model_name]['predictions']
            targets = self.results[model_name]['targets']
            errors = predictions - targets
            
            # Histogram
            axes[idx].hist(errors, bins=50, color=color, alpha=0.7, edgecolor='black')
            axes[idx].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
            axes[idx].axvline(x=errors.mean(), color='green', linestyle='--', linewidth=2,
                            label=f'Mean: {errors.mean():.4f}')
            
            axes[idx].set_xlabel('Prediction Error (rad)', fontsize=11)
            axes[idx].set_ylabel('Frequency', fontsize=11)
            axes[idx].set_title(f'{model_name}\nStd={np.std(errors):.4f}', 
                               fontsize=12, fontweight='bold')
            axes[idx].legend(fontsize=10)
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Error Distributions - Three Models', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = self.output_dir / 'error_distributions.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved error distributions: {save_path}")
        plt.close()
    
    def analyze_steering_range_performance(self):
        """Analyze performance across different steering ranges."""
        print(f"\n{'='*80}")
        print("STEERING RANGE PERFORMANCE ANALYSIS")
        print(f"{'='*80}\n")
        
        ranges = {
            'Straight (-0.05 to 0.05)': (-0.05, 0.05),
            'Slight Turns (0.05 to 0.15)': (0.05, 0.15),
            'Sharp Turns (> 0.15)': (0.15, float('inf')),
            'Left Turns (< -0.05)': (float('-inf'), -0.05)
        }
        
        range_results = {}
        
        for model_name in self.results.keys():
            predictions = self.results[model_name]['predictions']
            targets = self.results[model_name]['targets']
            
            print(f"{model_name}:")
            range_results[model_name] = {}
            
            for range_name, (lower, upper) in ranges.items():
                if range_name == 'Left Turns (< -0.05)':
                    mask = targets < upper
                elif range_name == 'Sharp Turns (> 0.15)':
                    mask = np.abs(targets) > lower
                else:
                    mask = (targets >= lower) & (targets < upper)
                
                if mask.sum() > 0:
                    range_targets = targets[mask]
                    range_predictions = predictions[mask]
                    mae = np.mean(np.abs(range_predictions - range_targets))
                    
                    range_results[model_name][range_name] = {
                        'n_samples': int(mask.sum()),
                        'mae': float(mae)
                    }
                    
                    print(f"  {range_name:30s}: MAE={mae:.4f} ({mask.sum()} samples)")
            print()
        
        # Save to JSON
        json_path = self.output_dir / 'range_analysis.json'
        with open(json_path, 'w') as f:
            json.dump(range_results, f, indent=4)
        print(f"✓ Saved range analysis: {json_path}\n")
        
        return range_results
    
    def create_final_report_figure(self):
        """Create comprehensive figure for final report."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        model_names = list(self.results.keys())
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        # Row 1: Scatter plots
        for idx, (model_name, color) in enumerate(zip(model_names, colors)):
            ax = fig.add_subplot(gs[0, idx])
            predictions = self.results[model_name]['predictions']
            targets = self.results[model_name]['targets']
            
            ax.scatter(targets, predictions, alpha=0.4, s=5, color=color)
            min_val = min(targets.min(), predictions.min())
            max_val = max(targets.max(), predictions.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            mae = self.results[model_name]['metrics']['mae']
            r2 = self.results[model_name]['metrics']['r2_score']
            ax.set_title(f'{model_name}\nMAE={mae:.4f}, R²={r2:.4f}', 
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('True (rad)', fontsize=10)
            ax.set_ylabel('Predicted (rad)', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Summary statistics in row 1, column 4
        ax_summary = fig.add_subplot(gs[0, 3])
        ax_summary.axis('off')
        
        summary_text = "Performance Summary\n" + "="*30 + "\n\n"
        for model_name in model_names:
            metrics = self.results[model_name]['metrics']
            summary_text += f"{model_name}:\n"
            summary_text += f"  MAE: {metrics['mae']:.4f} rad\n"
            summary_text += f"  MAE: {metrics['mae']*57.3:.2f}°\n"
            summary_text += f"  Acc±3°: {metrics['accuracy_5deg']:.1f}%\n"
            summary_text += f"  R²: {metrics['r2_score']:.4f}\n\n"
        
        ax_summary.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                       verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # Row 2: Error histograms
        for idx, (model_name, color) in enumerate(zip(model_names, colors)):
            ax = fig.add_subplot(gs[1, idx])
            predictions = self.results[model_name]['predictions']
            targets = self.results[model_name]['targets']
            errors = predictions - targets
            
            ax.hist(errors, bins=40, color=color, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax.set_title(f'{model_name} Errors\nStd={np.std(errors):.4f}', 
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Error (rad)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
        
        # Bar chart comparison in row 2, column 4
        ax_bars = fig.add_subplot(gs[1, 3])
        mae_values = [self.results[name]['metrics']['mae'] for name in model_names]
        bars = ax_bars.bar(range(len(model_names)), mae_values, color=colors, 
                          alpha=0.8, edgecolor='black', linewidth=2)
        ax_bars.set_xticks(range(len(model_names)))
        ax_bars.set_xticklabels(model_names, rotation=0)
        ax_bars.set_ylabel('MAE (rad)', fontsize=11)
        ax_bars.set_title('MAE Comparison', fontsize=12, fontweight='bold')
        ax_bars.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, mae_values):
            height = bar.get_height()
            ax_bars.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Row 3: Accuracy comparison
        ax_acc = fig.add_subplot(gs[2, :2])
        acc3_values = [self.results[name]['metrics']['accuracy_5deg'] for name in model_names]
        acc6_values = [self.results[name]['metrics']['accuracy_10deg'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax_acc.bar(x - width/2, acc3_values, width, label='±3°', 
                          color='steelblue', alpha=0.8, edgecolor='black')
        bars2 = ax_acc.bar(x + width/2, acc6_values, width, label='±6°', 
                          color='coral', alpha=0.8, edgecolor='black')
        
        ax_acc.set_ylabel('Accuracy (%)', fontsize=11)
        ax_acc.set_title('Accuracy Thresholds Comparison', fontsize=12, fontweight='bold')
        ax_acc.set_xticks(x)
        ax_acc.set_xticklabels(model_names)
        ax_acc.legend(fontsize=10)
        ax_acc.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax_acc.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # R² comparison in row 3, right side
        ax_r2 = fig.add_subplot(gs[2, 2:])
        r2_values = [self.results[name]['metrics']['r2_score'] for name in model_names]
        bars = ax_r2.bar(model_names, r2_values, color=colors, alpha=0.8, 
                        edgecolor='black', linewidth=2)
        ax_r2.set_ylabel('R² Score', fontsize=11)
        ax_r2.set_title('R² Score Comparison', fontsize=12, fontweight='bold')
        ax_r2.grid(True, alpha=0.3, axis='y')
        ax_r2.set_ylim([min(r2_values) * 0.95, 1.0])
        
        for bar, val in zip(bars, r2_values):
            height = bar.get_height()
            ax_r2.text(bar.get_x() + bar.get_width()/2., height,
                      f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.suptitle('Comprehensive Three-Model Comparison: Real vs CARLA vs Hybrid', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        save_path = self.output_dir / 'final_report_comprehensive_figure.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comprehensive figure: {save_path}")
        plt.close()
    
    def save_summary_report(self):
        """Save comprehensive summary to text file."""
        report_path = self.output_dir / 'comparison_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("THREE-MODEL COMPARISON SUMMARY\n")
            f.write("Real-World (TuSimple) vs CARLA (Synthetic) vs Hybrid\n")
            f.write("="*80 + "\n\n")
            
            for model_name in self.results.keys():
                metrics = self.results[model_name]['metrics']
                f.write(f"{model_name}:\n")
                f.write(f"  Samples: {metrics['n_samples']}\n")
                f.write(f"  MAE: {metrics['mae']:.4f} rad ({metrics['mae']*57.3:.2f}°)\n")
                f.write(f"  RMSE: {metrics['rmse']:.4f} rad\n")
                f.write(f"  R² Score: {metrics['r2_score']:.4f}\n")
                f.write(f"  Accuracy ±3°: {metrics['accuracy_5deg']:.1f}%\n")
                f.write(f"  Accuracy ±6°: {metrics['accuracy_10deg']:.1f}%\n")
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("KEY FINDINGS:\n")
            f.write("="*80 + "\n\n")
            
            # Find best model by MAE
            best_mae_model = min(self.results.items(), 
                               key=lambda x: x[1]['metrics']['mae'])
            f.write(f"Best MAE: {best_mae_model[0]} "
                   f"({best_mae_model[1]['metrics']['mae']:.4f} rad)\n")
            
            # Find best model by accuracy
            best_acc_model = max(self.results.items(), 
                               key=lambda x: x[1]['metrics']['accuracy_5deg'])
            f.write(f"Best Accuracy: {best_acc_model[0]} "
                   f"({best_acc_model[1]['metrics']['accuracy_5deg']:.1f}%)\n")
            
            f.write("\n")
        
        print(f"✓ Saved summary report: {report_path}")


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(
        description='Compare Real, CARLA, and Hybrid models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--real_checkpoint', type=str,
                       default='./checkpoints/pilotnet_tusimple_*/best_model.pth',
                       help='Path to real-world model checkpoint')
    parser.add_argument('--carla_checkpoint', type=str,
                       default='./checkpoints/pilotnet_carla_*/best_model.pth',
                       help='Path to CARLA model checkpoint')
    parser.add_argument('--hybrid_checkpoint', type=str,
                       default='./checkpoints/pilotnet_hybrid_*/best_model.pth',
                       help='Path to hybrid model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("THREE-MODEL COMPREHENSIVE COMPARISON")
    print(f"{'='*80}\n")
    
    # Initialize comparator
    comparator = ThreeModelComparison(args.device)
    
    # Find checkpoint files (handle wildcards)
    import glob
    
    real_ckpt = glob.glob(args.real_checkpoint)
    carla_ckpt = glob.glob(args.carla_checkpoint)
    hybrid_ckpt = glob.glob(args.hybrid_checkpoint)
    
    if not real_ckpt:
        print(f"✗ Real model checkpoint not found: {args.real_checkpoint}")
        return
    if not carla_ckpt:
        print(f"✗ CARLA model checkpoint not found: {args.carla_checkpoint}")
        return
    if not hybrid_ckpt:
        print(f"✗ Hybrid model checkpoint not found: {args.hybrid_checkpoint}")
        return
    
    real_ckpt = real_ckpt[0]
    carla_ckpt = carla_ckpt[0]
    hybrid_ckpt = hybrid_ckpt[0]
    
    # Load models
    real_model = comparator.load_model(real_ckpt, 'Real (TuSimple)')
    carla_model = comparator.load_model(carla_ckpt, 'CARLA (Synthetic)')
    hybrid_model = comparator.load_model(hybrid_ckpt, 'Hybrid')
    
    if not all([real_model, carla_model, hybrid_model]):
        print("✗ Failed to load all models")
        return
    
    # Load test data (use TuSimple validation set as common ground truth)
    print("\nLoading test data...")
    dataloaders = create_dataloaders(
        real_data_dir='./data/processed/tusimple_images',
        real_csv='./data/processed/tusimple_train_steering.csv',
        batch_size=args.batch_size,
        val_split=0.2,
        num_workers=0,
        training_mode='real'
    )
    test_loader = dataloaders['val']
    print(f"✓ Loaded {len(test_loader.dataset)} test samples\n")
    
    # Evaluate all models
    comparator.evaluate_model(real_model, test_loader, 'Real (TuSimple)')
    comparator.evaluate_model(carla_model, test_loader, 'CARLA (Synthetic)')
    comparator.evaluate_model(hybrid_model, test_loader, 'Hybrid')
    
    # Create visualizations
    print(f"\n{'='*80}")
    print("CREATING COMPARISON VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    comparator.create_comparison_table()
    comparator.plot_metrics_comparison()
    comparator.plot_prediction_scatter()
    comparator.plot_error_distributions()
    comparator.analyze_steering_range_performance()
    comparator.create_final_report_figure()
    comparator.save_summary_report()
    
    print(f"\n{'='*80}")
    print("✓ COMPARISON COMPLETE!")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {comparator.output_dir}")
    print("\nGenerated files:")
    print("  - model_comparison_table.csv")
    print("  - metrics_comparison.png")
    print("  - prediction_scatter_comparison.png")
    print("  - error_distributions.png")
    print("  - final_report_comprehensive_figure.png")
    print("  - comparison_summary.txt")
    print("  - range_analysis.json")
    print("\nUse these visualizations in your final report!")


if __name__ == "__main__":
    main()
