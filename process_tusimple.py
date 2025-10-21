"""
Process TuSimple Dataset
Convert lane annotations to steering angles

Author: Kip Chemweno
Course: ECE 4424 - Machine Learning
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil
import cv2


class TuSimpleProcessor:
    """Process TuSimple dataset - converts lane points to steering angles."""

    def __init__(self, dataset_path: str, output_dir: str = './data/processed'):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.output_images_dir = self.output_dir / 'tusimple_images'
        self.output_images_dir.mkdir(exist_ok=True)

    def calculate_steering_from_lanes(self, lanes, h_samples, image_width=1280, image_height=720):
        """
        Calculate steering angle from TuSimple lane points.

        Args:
            lanes: List of lane x-coordinates (one list per lane)
            h_samples: List of y-coordinates where lanes are sampled
            image_width: Image width (1280 for TuSimple)
            image_height: Image height (720 for TuSimple)

        Returns:
            Steering angle in radians, or None if invalid
        """
        if not lanes or not h_samples:
            return None

        # Focus on bottom 30% of image (immediate road ahead)
        target_y = int(image_height * 0.7)  # y=504 for 720p

        # Find closest h_sample to our target y
        h_idx = min(range(len(h_samples)),
                   key=lambda i: abs(h_samples[i] - target_y))

        # Get valid lane x positions at this y coordinate
        # -2 in TuSimple means no lane point at that position
        lane_x_positions = []
        for lane in lanes:
            if h_idx < len(lane) and lane[h_idx] != -2:
                lane_x_positions.append(lane[h_idx])

        if len(lane_x_positions) == 0:
            return None

        # Calculate lane center (mean of all visible lanes)
        lane_center = np.mean(lane_x_positions)
        image_center = image_width / 2.0  # 640 for 1280 width

        # Calculate horizontal offset in pixels
        offset_pixels = lane_center - image_center

        # Normalize to [-1, 1]
        normalized_offset = offset_pixels / (image_width / 2.0)

        # Convert to steering angle
        # Max steering: ±25 degrees = ±0.436 radians
        max_steering = 0.436
        steering_angle = -normalized_offset * max_steering  # Negative because left is negative

        # Clamp to valid range
        steering_angle = np.clip(steering_angle, -max_steering, max_steering)

        return float(steering_angle)

    def process_json_file(self, json_file: Path, base_path: Path):
        """Process a single JSON label file."""
        print(f"\n  Processing: {json_file.name}")

        if not json_file.exists():
            print(f"    ✗ File not found: {json_file}")
            return []

        data_records = []
        skipped = 0

        with open(json_file, 'r') as f:
            lines = f.readlines()

        print(f"    Found {len(lines)} annotations")

        for line in tqdm(lines, desc=f"    {json_file.name}", leave=False):
            try:
                label = json.loads(line.strip())

                # Extract data
                lanes = label['lanes']
                h_samples = label['h_samples']
                raw_file = label['raw_file']  # e.g., 'clips/0313-1/10/20.jpg'

                # Calculate steering angle
                steering = self.calculate_steering_from_lanes(lanes, h_samples)

                if steering is None:
                    skipped += 1
                    continue

                # Build full image path
                img_path = base_path / raw_file

                if not img_path.exists():
                    skipped += 1
                    continue

                # Create unique filename (replace slashes with underscores)
                unique_name = raw_file.replace('/', '_').replace('\\', '_')

                # Copy image to output directory
                output_img_path = self.output_images_dir / unique_name
                shutil.copy(img_path, output_img_path)

                data_records.append({
                    'image_path': unique_name,
                    'steering_angle': steering,
                    'original_path': raw_file
                })

            except Exception as e:
                skipped += 1
                continue

        print(f"    ✓ Processed: {len(data_records)}, Skipped: {skipped}")
        return data_records

    def process_train_set(self):
        """Process TuSimple training set."""
        print("\n" + "="*70)
        print("PROCESSING TUSIMPLE TRAINING SET")
        print("="*70)

        train_path = self.dataset_path / 'train_set'

        if not train_path.exists():
            print(f"✗ Training path not found: {train_path}")
            print(f"  Looking in: {self.dataset_path}")
            return None

        print(f"✓ Found training set at: {train_path}")

        # Process each label file
        label_files = [
            train_path / 'label_data_0313.json',
            train_path / 'label_data_0531.json',
            train_path / 'label_data_0601.json'
        ]

        all_records = []

        for label_file in label_files:
            if label_file.exists():
                records = self.process_json_file(label_file, train_path)
                all_records.extend(records)
            else:
                print(f"  ⚠️  Not found: {label_file.name}")

        if not all_records:
            print("\n✗ No valid samples processed!")
            return None

        # Create DataFrame
        df = pd.DataFrame(all_records)

        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split into train (80%) and validation (20%)
        train_size = int(0.8 * len(df))
        train_df = df[:train_size]
        val_df = df[train_size:]

        # Save CSVs (only image_path and steering_angle columns)
        train_csv = self.output_dir / 'tusimple_train_steering.csv'
        val_csv = self.output_dir / 'tusimple_val_steering.csv'

        train_df[['image_path', 'steering_angle']].to_csv(train_csv, index=False)
        val_df[['image_path', 'steering_angle']].to_csv(val_csv, index=False)

        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"✓ Total samples processed: {len(df)}")
        print(f"✓ Training samples: {len(train_df)}")
        print(f"✓ Validation samples: {len(val_df)}")
        print(f"\n✓ Train CSV: {train_csv}")
        print(f"✓ Val CSV: {val_csv}")
        print(f"✓ Images: {self.output_images_dir}")

        # Print statistics
        self._print_statistics(train_df, "TRAINING")
        self._print_statistics(val_df, "VALIDATION")

        return train_df, val_df

    def _print_statistics(self, df: pd.DataFrame, split_name: str):
        """Print dataset statistics."""
        print(f"\n{'-'*70}")
        print(f"{split_name} SET STATISTICS")
        print(f"{'-'*70}")
        print(f"Samples: {len(df)}")

        steering = df['steering_angle']

        print(f"\nSteering Angle:")
        print(f"  Mean:   {steering.mean():>8.4f} rad  ({np.degrees(steering.mean()):>6.2f}°)")
        print(f"  Median: {steering.median():>8.4f} rad  ({np.degrees(steering.median()):>6.2f}°)")
        print(f"  Std:    {steering.std():>8.4f} rad  ({np.degrees(steering.std()):>6.2f}°)")
        print(f"  Min:    {steering.min():>8.4f} rad  ({np.degrees(steering.min()):>6.2f}°)")
        print(f"  Max:    {steering.max():>8.4f} rad  ({np.degrees(steering.max()):>6.2f}°)")

        # Distribution analysis
        left = (steering < -0.05).sum()
        straight = ((steering >= -0.05) & (steering <= 0.05)).sum()
        right = (steering > 0.05).sum()

        total = len(df)
        print(f"\nSteering Distribution:")
        print(f"  Left turns  (< -0.05 rad): {left:>5d}  ({left/total*100:>5.1f}%)")
        print(f"  Straight (-0.05 to 0.05):  {straight:>5d}  ({straight/total*100:>5.1f}%)")
        print(f"  Right turns (> 0.05 rad):  {right:>5d}  ({right/total*100:>5.1f}%)")

        # Check for bias
        if straight / total > 0.75:
            print(f"\n  ⚠️  Dataset is {straight/total*100:.1f}% straight driving")
            print(f"     Consider using data augmentation (horizontal flip)")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Process TuSimple dataset')
    parser.add_argument('--path', type=str, default='./data/kaggle',
                       help='Path to TuSimple dataset (folder containing train_set/)')
    parser.add_argument('--output', type=str, default='./data/processed',
                       help='Output directory for processed data')

    args = parser.parse_args()

    print("="*70)
    print("TUSIMPLE DATASET PROCESSOR")
    print("="*70)
    print(f"Input:  {args.path}")
    print(f"Output: {args.output}")

    processor = TuSimpleProcessor(args.path, args.output)
    result = processor.process_train_set()

    if result:
        print("\n" + "="*70)
        print("PROCESSING COMPLETE! ✓")
        print("="*70)
        print("\nNext steps:")
        print("  1. Run EDA:")
        print("     python -c \"from eda_analysis import DrivingDataEDA; eda = DrivingDataEDA('./data/processed/tusimple_images', './data/processed/tusimple_train_steering.csv', 'tusimple'); eda.generate_full_report('./eda_results')\"")
        print("\n  2. Train model:")
        print("     python train.py --data_source tusimple --epochs 30 --batch_size 32")
        print("\n  3. Monitor training:")
        print("     tensorboard --logdir=./logs")
    else:
        print("\n" + "="*70)
        print("PROCESSING FAILED ✗")
        print("="*70)
        print("\nCheck that your dataset is in: ./data/kaggle/train_set/")


if __name__ == "__main__":
    main()