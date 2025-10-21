"""
Exploratory Data Analysis for Driving Dataset
Analyzes steering angle distribution, image quality, and data characteristics

Author: Kip Chemweno
Course: ECE 4424 - Machine Learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class DrivingDataEDA:
    """Comprehensive EDA for driving datasets."""
    
    def __init__(self, data_dir: str, csv_file: str, data_source: str = 'real'):
        """
        Initialize EDA analyzer.
        
        Args:
            data_dir: Directory containing images
            csv_file: Path to CSV with image paths and steering angles
            data_source: 'real', 'carla', or 'hybrid'
        """
        self.data_dir = Path(data_dir)
        self.csv_file = csv_file
        self.data_source = data_source
        
        # Load data
        self.df = pd.read_csv(csv_file)
        print(f"Loaded {len(self.df)} samples from {data_source} dataset")
        
    def basic_statistics(self):
        """Print basic dataset statistics."""
        print("\n" + "="*60)
        print("BASIC DATASET STATISTICS")
        print("="*60)
        
        print(f"\nTotal samples: {len(self.df):,}")
        print(f"\nColumns: {list(self.df.columns)}")
        print(f"\nData types:\n{self.df.dtypes}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        
        # Steering angle statistics
        steering = self.df['steering_angle']
        print(f"\n--- Steering Angle Statistics ---")
        print(f"Mean: {steering.mean():.4f}")
        print(f"Median: {steering.median():.4f}")
        print(f"Std Dev: {steering.std():.4f}")
        print(f"Min: {steering.min():.4f}")
        print(f"Max: {steering.max():.4f}")
        print(f"Range: {steering.max() - steering.min():.4f}")
        
        # Quantiles
        print(f"\nQuantiles:")
        for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
            print(f"  {int(q*100)}th percentile: {steering.quantile(q):.4f}")
    
    def plot_steering_distribution(self, save_path: str = None):
        """Plot steering angle distribution."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        steering = self.df['steering_angle']
        
        # Histogram
        axes[0, 0].hist(steering, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(steering.mean(), color='r', linestyle='--', 
                           label=f'Mean: {steering.mean():.3f}')
        axes[0, 0].axvline(steering.median(), color='g', linestyle='--', 
                           label=f'Median: {steering.median():.3f}')
        axes[0, 0].set_xlabel('Steering Angle')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title(f'Steering Angle Distribution ({self.data_source})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # KDE plot
        steering.plot(kind='kde', ax=axes[0, 1], linewidth=2)
        axes[0, 1].set_xlabel('Steering Angle')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Kernel Density Estimate')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot
        axes[1, 0].boxplot(steering, vert=True)
        axes[1, 0].set_ylabel('Steering Angle')
        axes[1, 0].set_title('Box Plot (Outlier Detection)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_steering = np.sort(steering)
        cumulative = np.arange(1, len(sorted_steering) + 1) / len(sorted_steering)
        axes[1, 1].plot(sorted_steering, cumulative, linewidth=2)
        axes[1, 1].set_xlabel('Steering Angle')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Distribution Function')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def analyze_steering_bias(self):
        """Analyze bias in steering angles."""
        steering = self.df['steering_angle']
        
        # Count samples in different ranges
        left = (steering < -0.1).sum()
        straight = ((steering >= -0.1) & (steering <= 0.1)).sum()
        right = (steering > 0.1).sum()
        
        total = len(steering)
        
        print("\n" + "="*60)
        print("STEERING BIAS ANALYSIS")
        print("="*60)
        print(f"Left turns (< -0.1):   {left:6d} ({left/total*100:.1f}%)")
        print(f"Straight (-0.1 to 0.1): {straight:6d} ({straight/total*100:.1f}%)")
        print(f"Right turns (> 0.1):   {right:6d} ({right/total*100:.1f}%)")
        
        # Check for severe imbalance
        if straight / total > 0.7:
            print("\n⚠️  WARNING: Dataset heavily biased toward straight driving!")
            print("   Consider data augmentation or balancing techniques.")
        
        return {'left': left, 'straight': straight, 'right': right}
    
    def sample_images(self, n_samples: int = 9, save_path: str = None):
        """Display random sample images with steering angles."""
        # Sample random indices
        sample_indices = np.random.choice(len(self.df), n_samples, replace=False)
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        for idx, sample_idx in enumerate(sample_indices):
            img_path = self.data_dir / self.df.iloc[sample_idx]['image_path']
            steering = self.df.iloc[sample_idx]['steering_angle']
            
            # Load image
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[idx].imshow(img)
                axes[idx].set_title(f'Steering: {steering:.3f}', fontsize=10)
                axes[idx].axis('off')
            else:
                axes[idx].text(0.5, 0.5, 'Image not found', 
                              ha='center', va='center')
                axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved sample images to {save_path}")
        
        plt.show()
    
    def analyze_image_properties(self, n_samples: int = 100):
        """Analyze image properties (size, brightness, etc.)."""
        print("\n" + "="*60)
        print("IMAGE PROPERTIES ANALYSIS")
        print("="*60)
        
        sample_indices = np.random.choice(len(self.df), 
                                         min(n_samples, len(self.df)), 
                                         replace=False)
        
        sizes = []
        brightnesses = []
        
        for idx in sample_indices:
            img_path = self.data_dir / self.df.iloc[idx]['image_path']
            img = cv2.imread(str(img_path))
            
            if img is not None:
                sizes.append(img.shape[:2])  # (height, width)
                # Calculate average brightness
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                brightnesses.append(gray.mean())
        
        # Analyze sizes
        unique_sizes = set(sizes)
        print(f"\nImage dimensions found: {len(unique_sizes)}")
        for size in unique_sizes:
            count = sizes.count(size)
            print(f"  {size[0]}x{size[1]}: {count} images ({count/len(sizes)*100:.1f}%)")
        
        # Analyze brightness
        if brightnesses:
            brightnesses = np.array(brightnesses)
            print(f"\nBrightness statistics:")
            print(f"  Mean: {brightnesses.mean():.2f}")
            print(f"  Std: {brightnesses.std():.2f}")
            print(f"  Min: {brightnesses.min():.2f}")
            print(f"  Max: {brightnesses.max():.2f}")
    
    def plot_steering_over_time(self, save_path: str = None):
        """Plot steering angle over sequence (if temporal data)."""
        fig, ax = plt.subplots(figsize=(15, 5))
        
        # Plot first 1000 samples or all if less
        n_plot = min(1000, len(self.df))
        steering = self.df['steering_angle'].iloc[:n_plot]
        
        ax.plot(steering, linewidth=0.5, alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Steering Angle')
        ax.set_title(f'Steering Angle Sequence (First {n_plot} samples)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def generate_full_report(self, output_dir: str = './eda_results'):
        """Generate complete EDA report with all visualizations."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"GENERATING FULL EDA REPORT")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")
        
        # Basic statistics
        self.basic_statistics()
        
        # Steering bias
        self.analyze_steering_bias()
        
        # Image properties
        self.analyze_image_properties()
        
        # Generate plots
        print("\nGenerating visualizations...")
        
        self.plot_steering_distribution(
            save_path=output_dir / f'steering_distribution_{self.data_source}.png'
        )
        
        self.sample_images(
            save_path=output_dir / f'sample_images_{self.data_source}.png'
        )
        
        self.plot_steering_over_time(
            save_path=output_dir / f'steering_sequence_{self.data_source}.png'
        )
        
        print(f"\n✓ EDA report completed! Results saved to {output_dir}")


def compare_datasets(real_eda: DrivingDataEDA, carla_eda: DrivingDataEDA, 
                     save_path: str = './eda_results/comparison.png'):
    """Compare real-world and CARLA datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Real-world distribution
    axes[0].hist(real_eda.df['steering_angle'], bins=50, 
                alpha=0.7, label='Real-world', edgecolor='black')
    axes[0].set_xlabel('Steering Angle')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Real-world Data Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # CARLA distribution
    axes[1].hist(carla_eda.df['steering_angle'], bins=50, 
                alpha=0.7, label='CARLA', color='orange', edgecolor='black')
    axes[1].set_xlabel('Steering Angle')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('CARLA Data Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {save_path}")
    plt.show()
    
    # Statistical comparison
    print("\n" + "="*60)
    print("DATASET COMPARISON")
    print("="*60)
    
    real_steering = real_eda.df['steering_angle']
    carla_steering = carla_eda.df['steering_angle']
    
    print(f"\n{'Metric':<20} {'Real-world':<15} {'CARLA':<15} {'Difference'}")
    print("-" * 60)
    print(f"{'Sample count':<20} {len(real_steering):<15,} {len(carla_steering):<15,} {len(real_steering) - len(carla_steering):,}")
    print(f"{'Mean':<20} {real_steering.mean():<15.4f} {carla_steering.mean():<15.4f} {real_steering.mean() - carla_steering.mean():.4f}")
    print(f"{'Std Dev':<20} {real_steering.std():<15.4f} {carla_steering.std():<15.4f} {real_steering.std() - carla_steering.std():.4f}")
    print(f"{'Min':<20} {real_steering.min():<15.4f} {carla_steering.min():<15.4f} {real_steering.min() - carla_steering.min():.4f}")
    print(f"{'Max':<20} {real_steering.max():<15.4f} {carla_steering.max():<15.4f} {real_steering.max() - carla_steering.max():.4f}")


if __name__ == "__main__":
    print("EDA module loaded successfully")
    print("\nExample usage:")
    print("  eda = DrivingDataEDA('data/images', 'data/steering.csv', 'real')")
    print("  eda.generate_full_report('./eda_results')")
