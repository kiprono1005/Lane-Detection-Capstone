"""
Convert Lane Segmentation Labels to Steering Angles
Processes Kaggle and Roboflow datasets

Author: Kip Chemweno
Course: ECE 4424 - Machine Learning
"""

import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import json
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil


class LaneToSteering:
    """Convert lane segmentation masks to steering angles."""

    def __init__(self, image_width=1080, image_height=640):
        self.image_width = image_width
        self.image_height = image_height

    def calculate_steering_from_mask(self, lane_mask: np.ndarray,
                                     visualize: bool = False) -> Optional[float]:
        """
        Calculate steering angle from lane segmentation mask.

        Args:
            lane_mask: Binary mask where lanes are marked (255 = lane, 0 = background)
            visualize: Whether to show the calculation

        Returns:
            Steering angle in radians (negative=left, positive=right)
        """
        height, width = lane_mask.shape[:2]

        # Focus on bottom 30% of image (immediate road ahead)
        roi_height = int(height * 0.3)
        roi_start = height - roi_height
        roi = lane_mask[roi_start:, :]

        # Find lane pixels in ROI
        lane_pixels = np.where(roi > 127)  # Threshold for lane pixels

        if len(lane_pixels[0]) < 10:  # Need minimum pixels
            return None

        # Calculate lane center (mean x position of lane pixels)
        lane_center_x = np.mean(lane_pixels[1])
        image_center_x = width / 2.0

        # Calculate offset from center (in pixels)
        offset_pixels = lane_center_x - image_center_x

        # Normalize offset to [-1, 1]
        max_offset = width / 2.0
        normalized_offset = offset_pixels / max_offset

        # Convert to steering angle
        # Typical max steering: ±25 degrees = ±0.436 radians
        max_steering_angle = 0.436
        steering_angle = -normalized_offset * max_steering_angle

        # Clamp to valid range
        steering_angle = np.clip(steering_angle, -max_steering_angle, max_steering_angle)

        if visualize:
            self._visualize_steering(lane_mask, roi_start, lane_center_x,
                                     image_center_x, steering_angle)

        return float(steering_angle)

    def calculate_steering_from_boundaries(self, left_mask: np.ndarray,
                                           right_mask: np.ndarray,
                                           visualize: bool = False) -> Optional[float]:
        """
        Calculate steering from separate left and right lane boundaries.
        More accurate than single combined mask.

        Args:
            left_mask: Binary mask of left lane boundary
            right_mask: Binary mask of right lane boundary

        Returns:
            Steering angle in radians
        """
        height, width = left_mask.shape[:2]
        roi_height = int(height * 0.3)
        roi_start = height - roi_height

        # Get ROI for both lanes
        left_roi = left_mask[roi_start:, :]
        right_roi = right_mask[roi_start:, :]

        # Find lane boundaries
        left_pixels = np.where(left_roi > 127)
        right_pixels = np.where(right_roi > 127)

        # Need both lanes visible
        if len(left_pixels[0]) < 10 or len(right_pixels[0]) < 10:
            # Fall back to whichever lane is visible
            if len(left_pixels[0]) >= 10:
                return self.calculate_steering_from_mask(left_mask, visualize)
            elif len(right_pixels[0]) >= 10:
                return self.calculate_steering_from_mask(right_mask, visualize)
            else:
                return None

        # Calculate center of each lane boundary
        left_x = np.mean(left_pixels[1])
        right_x = np.mean(right_pixels[1])

        # Lane center is midpoint between boundaries
        lane_center = (left_x + right_x) / 2.0
        image_center = width / 2.0

        # Calculate steering angle
        offset = lane_center - image_center
        normalized_offset = offset / (width / 2.0)
        steering_angle = -normalized_offset * 0.436

        steering_angle = np.clip(steering_angle, -0.436, 0.436)

        if visualize:
            combined_mask = cv2.addWeighted(left_mask, 0.5, right_mask, 0.5, 0)
            self._visualize_steering(combined_mask, roi_start, lane_center,
                                     image_center, steering_angle)

        return float(steering_angle)

    def _visualize_steering(self, mask, roi_start, lane_center,
                            image_center, steering_angle):
        """Visualize steering calculation."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        ax.imshow(mask, cmap='gray')

        # Draw ROI boundary
        ax.axhline(y=roi_start, color='r', linestyle='--', linewidth=2,
                   label='ROI boundary')

        # Draw centers
        ax.axvline(x=image_center, color='g', linestyle='--', linewidth=2,
                   label='Image center')
        ax.axvline(x=lane_center, color='b', linestyle='-', linewidth=2,
                   label='Lane center')

        # Draw arrow showing steering direction
        arrow_start = (image_center, roi_start + 50)
        arrow_end = (lane_center, roi_start + 50)
        ax.annotate('', xy=arrow_end, xytext=arrow_start,
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=3))

        # Add text info
        direction = "LEFT" if steering_angle < 0 else "RIGHT" if steering_angle > 0 else "STRAIGHT"
        ax.set_title(f'Steering: {steering_angle:.3f} rad ({np.degrees(steering_angle):.1f}°) - {direction}',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        plt.tight_layout()
        plt.show()


class KaggleDatasetProcessor:
    """Process Kaggle lane segmentation dataset."""

    def __init__(self, dataset_path: str, output_dir: str = './data/processed'):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.converter = LaneToSteering()

        # Create output images directory
        self.output_images_dir = self.output_dir / 'kaggle_images'
        self.output_images_dir.mkdir(exist_ok=True)

    def process_dataset(self, split: str = 'train', visualize_samples: int = 0):
        """
        Process Kaggle dataset split.

        Args:
            split: 'train' or 'val'
            visualize_samples: Number of samples to visualize (0 = none)
        """
        print(f"\n{'=' * 70}")
        print(f"PROCESSING KAGGLE {split.upper()} SPLIT")
        print(f"{'=' * 70}")

        images_dir = self.dataset_path / split
        labels_dir = self.dataset_path / f"{split}_label"

        if not images_dir.exists() or not labels_dir.exists():
            print(f"✗ Directory not found:")
            print(f"  Images: {images_dir}")
            print(f"  Labels: {labels_dir}")
            return None

        data_records = []
        image_files = sorted(images_dir.glob('*.png'))

        print(f"Found {len(image_files)} images")
        print("Processing...")

        visualized = 0

        for img_file in tqdm(image_files, desc=f"Processing {split}"):
            label_file = labels_dir / img_file.name

            if not label_file.exists():
                continue

            # Load label mask
            label_mask = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)

            if label_mask is None:
                continue

            # Kaggle labels: 0 = background, 1 = left lane, 2 = right lane
            left_mask = (label_mask == 1).astype(np.uint8) * 255
            right_mask = (label_mask == 2).astype(np.uint8) * 255

            # Visualize some samples
            show_viz = visualize_samples > 0 and visualized < visualize_samples

            # Calculate steering angle
            steering = self.converter.calculate_steering_from_boundaries(
                left_mask, right_mask, visualize=show_viz
            )

            if show_viz:
                visualized += 1

            if steering is not None:
                # Copy image to processed directory
                output_img_path = self.output_images_dir / img_file.name
                shutil.copy(img_file, output_img_path)

                data_records.append({
                    'image_path': img_file.name,
                    'steering_angle': steering
                })

        print(f"\n✓ Successfully processed {len(data_records)}/{len(image_files)} samples")

        if data_records:
            df = pd.DataFrame(data_records)
            csv_path = self.output_dir / f'kaggle_{split}_steering.csv'
            df.to_csv(csv_path, index=False)
            print(f"✓ Saved CSV to: {csv_path}")
            print(f"✓ Images copied to: {self.output_images_dir}")

            self._print_statistics(df, split)
            return df
        else:
            print("✗ No valid samples processed")
            return None

    def _print_statistics(self, df: pd.DataFrame, split: str):
        """Print dataset statistics."""
        print(f"\n{'=' * 70}")
        print(f"KAGGLE {split.upper()} STATISTICS")
        print(f"{'=' * 70}")
        print(f"Total samples: {len(df)}")
        print(f"\nSteering Angle Statistics:")
        print(f"  Mean:   {df['steering_angle'].mean():>8.4f} rad")
        print(f"  Median: {df['steering_angle'].median():>8.4f} rad")
        print(f"  Std:    {df['steering_angle'].std():>8.4f} rad")
        print(f"  Min:    {df['steering_angle'].min():>8.4f} rad ({np.degrees(df['steering_angle'].min()):>6.2f}°)")
        print(f"  Max:    {df['steering_angle'].max():>8.4f} rad ({np.degrees(df['steering_angle'].max()):>6.2f}°)")

        # Distribution analysis
        left_turns = (df['steering_angle'] < -0.05).sum()
        straight = ((df['steering_angle'] >= -0.05) & (df['steering_angle'] <= 0.05)).sum()
        right_turns = (df['steering_angle'] > 0.05).sum()

        print(f"\nSteering Distribution:")
        print(f"  Left turns  (<-0.05 rad): {left_turns:>6d} ({left_turns / len(df) * 100:>5.1f}%)")
        print(f"  Straight (-0.05 to 0.05): {straight:>6d} ({straight / len(df) * 100:>5.1f}%)")
        print(f"  Right turns (>0.05 rad):  {right_turns:>6d} ({right_turns / len(df) * 100:>5.1f}%)")

        if straight / len(df) > 0.7:
            print("\n⚠️  WARNING: Dataset heavily biased toward straight driving!")
            print("   Consider data augmentation or balancing techniques.")


class RoboflowDatasetProcessor:
    """Process Roboflow COCO format dataset."""

    def __init__(self, dataset_path: str, output_dir: str = './data/processed'):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.converter = LaneToSteering()

        self.output_images_dir = self.output_dir / 'roboflow_images'
        self.output_images_dir.mkdir(exist_ok=True)

    def process_dataset(self, split: str = 'train', visualize_samples: int = 0):
        """Process Roboflow dataset split."""
        print(f"\n{'=' * 70}")
        print(f"PROCESSING ROBOFLOW {split.upper()} SPLIT")
        print(f"{'=' * 70}")

        split_dir = self.dataset_path / split
        annotations_file = split_dir / '_annotations.coco.json'

        if not annotations_file.exists():
            print(f"✗ Annotations not found: {annotations_file}")
            return None

        # Load COCO annotations
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)

        print(f"Found {len(coco_data['images'])} images in annotations")

        # Create mappings
        images_dict = {img['id']: img for img in coco_data['images']}

        # Group annotations by image
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)

        data_records = []
        visualized = 0

        print("Processing...")

        for img_id, img_info in tqdm(images_dict.items(), desc=f"Processing {split}"):
            filename = img_info['file_name']
            img_path = split_dir / filename

            if not img_path.exists():
                continue

            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            height, width = img.shape[:2]

            # Create lane mask from COCO annotations
            lane_mask = np.zeros((height, width), dtype=np.uint8)

            if img_id in annotations_by_image:
                for ann in annotations_by_image[img_id]:
                    if 'segmentation' in ann and len(ann['segmentation']) > 0:
                        for seg in ann['segmentation']:
                            # Convert flat list to polygon points
                            points = np.array(seg).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(lane_mask, [points], 255)

            # Visualize some samples
            show_viz = visualize_samples > 0 and visualized < visualize_samples

            # Calculate steering
            steering = self.converter.calculate_steering_from_mask(
                lane_mask, visualize=show_viz
            )

            if show_viz:
                visualized += 1

            if steering is not None:
                # Copy image
                output_img_path = self.output_images_dir / filename
                shutil.copy(img_path, output_img_path)

                data_records.append({
                    'image_path': filename,
                    'steering_angle': steering
                })

        print(f"\n✓ Successfully processed {len(data_records)}/{len(images_dict)} samples")

        if data_records:
            df = pd.DataFrame(data_records)
            csv_path = self.output_dir / f'roboflow_{split}_steering.csv'
            df.to_csv(csv_path, index=False)
            print(f"✓ Saved CSV to: {csv_path}")
            print(f"✓ Images copied to: {self.output_images_dir}")

            self._print_statistics(df, split)
            return df
        else:
            print("✗ No valid samples processed")
            return None

    def _print_statistics(self, df: pd.DataFrame, split: str):
        """Print dataset statistics."""
        print(f"\n{'=' * 70}")
        print(f"ROBOFLOW {split.upper()} STATISTICS")
        print(f"{'=' * 70}")
        print(f"Total samples: {len(df)}")
        print(f"\nSteering Angle Statistics:")
        print(f"  Mean:   {df['steering_angle'].mean():>8.4f} rad")
        print(f"  Median: {df['steering_angle'].median():>8.4f} rad")
        print(f"  Std:    {df['steering_angle'].std():>8.4f} rad")
        print(f"  Min:    {df['steering_angle'].min():>8.4f} rad ({np.degrees(df['steering_angle'].min()):>6.2f}°)")
        print(f"  Max:    {df['steering_angle'].max():>8.4f} rad ({np.degrees(df['steering_angle'].max()):>6.2f}°)")

        left_turns = (df['steering_angle'] < -0.05).sum()
        straight = ((df['steering_angle'] >= -0.05) & (df['steering_angle'] <= 0.05)).sum()
        right_turns = (df['steering_angle'] > 0.05).sum()

        print(f"\nSteering Distribution:")
        print(f"  Left turns:  {left_turns:>6d} ({left_turns / len(df) * 100:>5.1f}%)")
        print(f"  Straight:    {straight:>6d} ({straight / len(df) * 100:>5.1f}%)")
        print(f"  Right turns: {right_turns:>6d} ({right_turns / len(df) * 100:>5.1f}%)")


def main():
    """Main processing function with CLI."""
    import argparse

    parser = argparse.ArgumentParser(description='Convert lane labels to steering angles')
    parser.add_argument('--dataset', required=True, choices=['kaggle', 'roboflow'],
                        help='Dataset type')
    parser.add_argument('--path', required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output', default='./data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--splits', nargs='+', default=['train'],
                        help='Splits to process (e.g., train val test)')
    parser.add_argument('--visualize', type=int, default=0,
                        help='Number of samples to visualize per split')

    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"LANE TO STEERING ANGLE CONVERTER")
    print(f"{'=' * 70}")
    print(f"Dataset: {args.dataset}")
    print(f"Input path: {args.path}")
    print(f"Output path: {args.output}")
    print(f"Splits: {args.splits}")

    if args.dataset == 'kaggle':
        processor = KaggleDatasetProcessor(args.path, args.output)
    else:
        processor = RoboflowDatasetProcessor(args.path, args.output)

    for split in args.splits:
        processor.process_dataset(split, visualize_samples=args.visualize)

    print(f"\n{'=' * 70}")
    print("PROCESSING COMPLETE!")
    print(f"{'=' * 70}")
    print("\nNext steps:")
    print("  1. Check the statistics above for data quality")
    print("  2. Run EDA on processed data")
    print(f"  3. Train model: python train.py --data_source {args.dataset}")


if __name__ == "__main__":
    main()