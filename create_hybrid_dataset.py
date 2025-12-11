"""
Create Hybrid Dataset - Three Experimental Modes
Supports rigorous scientific comparison of real vs synthetic data

Author: Kip Chemweno
Course: ECE 4424 - Machine Learning
Date: December 2025

Three Modes:
1. EQUAL (50-50): Fair comparison, proves synthetic adds value beyond quantity
2. REAL_HEAVY (70-30): Tests prioritizing real data with synthetic supplement
3. AUGMENT: Maximum performance, all real + all synthetic data

Usage Examples:
    # Experiment 1: Fair comparison (PRIMARY EXPERIMENT)
    python create_hybrid_dataset.py --mode equal --total_samples 3000

    # Experiment 2: Real-heavy mix
    python create_hybrid_dataset.py --mode real_heavy --total_samples 3000

    # Experiment 3: Full augmentation (PRACTICAL APPLICATION)
    python create_hybrid_dataset.py --mode augment
"""

import pandas as pd
import numpy as np
import shutil
from pathlib import Path
import argparse
import json
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class HybridDatasetCreator:
    """Create hybrid dataset with controlled experimental setup."""

    def __init__(self, real_dir: str, carla_dir: str, output_dir: str):
        self.real_dir = Path(real_dir)
        self.carla_dir = Path(carla_dir)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_images_dir = self.output_dir / 'hybrid_images'
        self.output_images_dir.mkdir(exist_ok=True)

    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load real and CARLA datasets."""
        print("\n" + "="*70)
        print("LOADING DATASETS")
        print("="*70)

        # Load TuSimple (real)
        real_train = self.real_dir / 'tusimple_train_steering.csv'
        real_val = self.real_dir / 'tusimple_val_steering.csv'

        if not real_train.exists():
            raise FileNotFoundError(f"TuSimple data not found: {real_train}")

        real_train_df = pd.read_csv(real_train)
        real_val_df = pd.read_csv(real_val) if real_val.exists() else pd.DataFrame()

        real_df = pd.concat([real_train_df, real_val_df], ignore_index=True)
        real_df['source'] = 'real'
        real_df['original_path'] = real_df['image_path'].copy()

        print(f"✓ Loaded TuSimple (Real): {len(real_df):,} samples")

        # Load CARLA (synthetic)
        carla_csv = self.carla_dir / 'carla_steering.csv'

        if not carla_csv.exists():
            raise FileNotFoundError(f"CARLA data not found: {carla_csv}")

        carla_df = pd.read_csv(carla_csv)
        carla_df['source'] = 'carla'
        carla_df['original_path'] = carla_df['image_path'].copy()

        print(f"✓ Loaded CARLA (Synthetic): {len(carla_df):,} samples")

        return real_df, carla_df

    def create_equal_mix(self, real_df: pd.DataFrame, carla_df: pd.DataFrame,
                        total_samples: int) -> pd.DataFrame:
        """
        Create 50-50 mix with controlled total size.

        Purpose: Fair comparison - proves synthetic adds value beyond quantity.
        """
        print("\n" + "="*70)
        print(f"CREATING EQUAL MIX (50-50 SPLIT)")
        print("="*70)
        print(f"Target total: {total_samples:,} samples")
        print(f"  Real: {total_samples//2:,} samples")
        print(f"  CARLA: {total_samples//2:,} samples")

        samples_per_source = total_samples // 2

        # Sample from each
        real_sample = real_df.sample(n=min(samples_per_source, len(real_df)),
                                    random_state=42)
        carla_sample = carla_df.sample(n=min(samples_per_source, len(carla_df)),
                                      random_state=42)

        combined = pd.concat([real_sample, carla_sample], ignore_index=True)

        print(f"\n✓ Created equal mix:")
        print(f"  Real: {len(real_sample):,} samples ({len(real_sample)/len(combined)*100:.1f}%)")
        print(f"  CARLA: {len(carla_sample):,} samples ({len(carla_sample)/len(combined)*100:.1f}%)")
        print(f"  Total: {len(combined):,} samples")

        return combined

    def create_real_heavy_mix(self, real_df: pd.DataFrame, carla_df: pd.DataFrame,
                             total_samples: int) -> pd.DataFrame:
        """
        Create 70-30 mix favoring real data.

        Purpose: Test if synthetic helps even when real is prioritized.
        """
        print("\n" + "="*70)
        print(f"CREATING REAL-HEAVY MIX (70-30 SPLIT)")
        print("="*70)
        print(f"Target total: {total_samples:,} samples")

        real_samples = int(total_samples * 0.7)
        carla_samples = total_samples - real_samples

        print(f"  Real: {real_samples:,} samples (70%)")
        print(f"  CARLA: {carla_samples:,} samples (30%)")

        real_sample = real_df.sample(n=min(real_samples, len(real_df)),
                                    random_state=42)
        carla_sample = carla_df.sample(n=min(carla_samples, len(carla_df)),
                                      random_state=42)

        combined = pd.concat([real_sample, carla_sample], ignore_index=True)

        print(f"\n✓ Created real-heavy mix:")
        print(f"  Real: {len(real_sample):,} samples ({len(real_sample)/len(combined)*100:.1f}%)")
        print(f"  CARLA: {len(carla_sample):,} samples ({len(carla_sample)/len(combined)*100:.1f}%)")
        print(f"  Total: {len(combined):,} samples")

        return combined

    def create_augment_mix(self, real_df: pd.DataFrame,
                          carla_df: pd.DataFrame) -> pd.DataFrame:
        """
        Use ALL available data from both sources.

        Purpose: Maximum performance - practical augmentation scenario.
        """
        print("\n" + "="*70)
        print(f"CREATING AUGMENTATION MIX (ALL DATA)")
        print("="*70)
        print(f"  Real: ALL {len(real_df):,} samples")
        print(f"  CARLA: ALL {len(carla_df):,} samples")

        combined = pd.concat([real_df, carla_df], ignore_index=True)

        print(f"\n✓ Created augmentation mix:")
        print(f"  Real: {len(real_df):,} samples ({len(real_df)/len(combined)*100:.1f}%)")
        print(f"  CARLA: {len(carla_df):,} samples ({len(carla_df)/len(combined)*100:.1f}%)")
        print(f"  Total: {len(combined):,} samples")

        return combined

    def analyze_distribution(self, df: pd.DataFrame, name: str):
        """Analyze steering distribution."""
        print(f"\n{name} Distribution:")

        steering = df['steering_angle'].values

        # Categories
        sharp_left = (steering < -0.15).sum()
        left = ((steering >= -0.15) & (steering < -0.05)).sum()
        straight = ((steering >= -0.05) & (steering <= 0.05)).sum()
        right = ((steering > 0.05) & (steering <= 0.15)).sum()
        sharp_right = (steering > 0.15).sum()

        total = len(steering)

        print(f"  Sharp Left (<-0.15):    {sharp_left:5d} ({sharp_left/total*100:5.1f}%)")
        print(f"  Left (-0.15 to -0.05):  {left:5d} ({left/total*100:5.1f}%)")
        print(f"  Straight (-0.05-0.05):  {straight:5d} ({straight/total*100:5.1f}%)")
        print(f"  Right (0.05 to 0.15):   {right:5d} ({right/total*100:5.1f}%)")
        print(f"  Sharp Right (>0.15):    {sharp_right:5d} ({sharp_right/total*100:5.1f}%)")

        print(f"\n  Mean: {steering.mean():.4f} rad ({np.degrees(steering.mean()):.2f}°)")
        print(f"  Std:  {steering.std():.4f} rad ({np.degrees(steering.std()):.2f}°)")

    def copy_images(self, df: pd.DataFrame):
        """Copy images to hybrid directory with source prefix."""
        print("\n" + "="*70)
        print("COPYING IMAGES")
        print("="*70)

        from tqdm import tqdm

        real_images_dir = self.real_dir / 'tusimple_images'
        carla_images_dir = self.carla_dir / 'images'

        new_paths = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Copying"):
            source = row['source']
            orig_path = row['original_path']

            # Determine source directory
            if source == 'real':
                src_img = real_images_dir / orig_path
                new_name = f"real_{orig_path}"
            else:  # carla
                src_img = carla_images_dir / orig_path
                new_name = f"carla_{orig_path}"

            dst_img = self.output_images_dir / new_name

            if src_img.exists():
                shutil.copy(src_img, dst_img)
                new_paths.append(new_name)
            else:
                print(f"  ⚠ Missing: {src_img}")
                new_paths.append(None)

        df['image_path'] = new_paths

        # Remove rows with missing images
        df = df[df['image_path'].notna()].reset_index(drop=True)

        print(f"✓ Copied {len(df):,} images")
        return df

    def create_train_val_split(self, df: pd.DataFrame,
                               val_split: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create stratified train/val split."""
        print("\n" + "="*70)
        print("CREATING TRAIN/VAL SPLIT")
        print("="*70)

        # Categorize for stratification
        def categorize(angle):
            if angle < -0.05:
                return 'left'
            elif angle > 0.05:
                return 'right'
            else:
                return 'straight'

        df['category'] = df['steering_angle'].apply(categorize)

        # Stratified split
        train_dfs = []
        val_dfs = []

        for category in df['category'].unique():
            cat_df = df[df['category'] == category]
            n_val = int(len(cat_df) * val_split)

            val_df = cat_df.sample(n=n_val, random_state=42)
            train_df = cat_df.drop(val_df.index)

            train_dfs.append(train_df)
            val_dfs.append(val_df)

        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)

        # Shuffle
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Drop temporary category column
        train_df = train_df.drop('category', axis=1)
        val_df = val_df.drop('category', axis=1)

        print(f"✓ Train: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
        print(f"✓ Val:   {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")

        return train_df, val_df

    def save_dataset(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                    mode: str):
        """Save hybrid dataset."""
        print("\n" + "="*70)
        print("SAVING DATASET")
        print("="*70)

        # Save CSVs (only needed columns)
        train_csv = self.output_dir / 'hybrid_train_steering.csv'
        val_csv = self.output_dir / 'hybrid_val_steering.csv'

        train_df[['image_path', 'steering_angle']].to_csv(train_csv, index=False)
        val_df[['image_path', 'steering_angle']].to_csv(val_csv, index=False)

        print(f"✓ Train CSV: {train_csv}")
        print(f"✓ Val CSV: {val_csv}")

        # Metadata
        combined_df = pd.concat([train_df, val_df])

        metadata = {
            'mode': mode,
            'total_samples': int(len(combined_df)),
            'train_samples': int(len(train_df)),
            'val_samples': int(len(val_df)),
            'real_samples': int((combined_df['source'] == 'real').sum()),
            'carla_samples': int((combined_df['source'] == 'carla').sum()),
            'real_percentage': float((combined_df['source'] == 'real').sum() / len(combined_df) * 100),
            'carla_percentage': float((combined_df['source'] == 'carla').sum() / len(combined_df) * 100),
            'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        metadata_path = self.output_dir / 'hybrid_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"✓ Metadata: {metadata_path}")

        return metadata

    def create_visualization(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Create distribution comparison plot."""
        combined_df = pd.concat([train_df, val_df])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # By source
        real_steering = combined_df[combined_df['source'] == 'real']['steering_angle']
        carla_steering = combined_df[combined_df['source'] == 'carla']['steering_angle']

        axes[0].hist(real_steering, bins=50, alpha=0.7, label='Real', edgecolor='black')
        axes[0].hist(carla_steering, bins=50, alpha=0.7, label='CARLA', edgecolor='black')
        axes[0].set_xlabel('Steering Angle (radians)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Hybrid Dataset Distribution by Source')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Combined
        axes[1].hist(combined_df['steering_angle'], bins=50,
                    alpha=0.7, color='green', edgecolor='black')
        axes[1].set_xlabel('Steering Angle (radians)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Combined Hybrid Distribution')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = self.output_dir / 'hybrid_distribution.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization: {plot_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create hybrid dataset with three experimental modes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experimental Modes:
  equal:       50-50 split (PRIMARY - proves synthetic value)
               Example: python create_hybrid_dataset.py --mode equal --total_samples 3000
               
  real_heavy:  70-30 split favoring real data
               Example: python create_hybrid_dataset.py --mode real_heavy --total_samples 3000
               
  augment:     ALL real + ALL synthetic (maximum performance)
               Example: python create_hybrid_dataset.py --mode augment

For Your Research:
  1. Start with 'equal' mode (most rigorous)
  2. Optionally try 'augment' to show practical benefit
  3. 'real_heavy' is optional exploratory experiment
        """
    )

    parser.add_argument('--real', type=str, default='./data/processed',
                       help='Path to processed TuSimple data')
    parser.add_argument('--carla', type=str, default='./data/carla',
                       help='Path to CARLA data')
    parser.add_argument('--output', type=str, default='./data/hybrid',
                       help='Output directory')
    parser.add_argument('--mode', type=str, default='equal',
                       choices=['equal', 'real_heavy', 'augment'],
                       help='Dataset creation mode (see examples above)')
    parser.add_argument('--total_samples', type=int, default=3000,
                       help='Total samples for equal/real_heavy modes (ignored for augment)')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("HYBRID DATASET CREATOR")
    print("="*70)
    print(f"Mode: {args.mode.upper()}")
    if args.mode != 'augment':
        print(f"Target samples: {args.total_samples:,}")
    print(f"Real data: {args.real}")
    print(f"CARLA data: {args.carla}")
    print(f"Output: {args.output}")

    # Explain the chosen mode
    if args.mode == 'equal':
        print("\n📊 EQUAL MODE (50-50 split)")
        print("  Purpose: Fair comparison - proves synthetic adds value")
        print("  Use for: Primary research result")
    elif args.mode == 'real_heavy':
        print("\n📊 REAL-HEAVY MODE (70-30 split)")
        print("  Purpose: Test if synthetic helps when real is prioritized")
        print("  Use for: Exploratory experiment")
    else:  # augment
        print("\n📊 AUGMENT MODE (all data)")
        print("  Purpose: Maximum performance - practical application")
        print("  Use for: Showing real-world benefit")

    print("="*70)

    creator = HybridDatasetCreator(args.real, args.carla, args.output)

    # Load data
    real_df, carla_df = creator.load_datasets()

    # Create mix based on mode
    if args.mode == 'equal':
        combined_df = creator.create_equal_mix(real_df, carla_df, args.total_samples)
    elif args.mode == 'real_heavy':
        combined_df = creator.create_real_heavy_mix(real_df, carla_df, args.total_samples)
    else:  # augment
        combined_df = creator.create_augment_mix(real_df, carla_df)

    # Analyze
    creator.analyze_distribution(combined_df, "Hybrid Dataset")

    # Copy images
    combined_df = creator.copy_images(combined_df)

    # Train/val split
    train_df, val_df = creator.create_train_val_split(combined_df, args.val_split)

    # Analyze splits
    creator.analyze_distribution(train_df, "Training Set")
    creator.analyze_distribution(val_df, "Validation Set")

    # Save
    metadata = creator.save_dataset(train_df, val_df, args.mode)

    # Visualize
    creator.create_visualization(train_df, val_df)

    print("\n" + "="*70)
    print("✓ HYBRID DATASET CREATED!")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Total: {metadata['total_samples']:,} samples")
    print(f"  Real: {metadata['real_samples']:,} ({metadata['real_percentage']:.1f}%)")
    print(f"  CARLA: {metadata['carla_samples']:,} ({metadata['carla_percentage']:.1f}%)")
    print(f"\nSaved to: {args.output}")
    print(f"\nNext step:")
    print(f"  python train.py --data_source hybrid --epochs 30 --num_workers 0")


if __name__ == "__main__":
    main()