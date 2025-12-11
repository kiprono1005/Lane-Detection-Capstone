"""
Autonomous Lane Keeping - Data Loading Pipeline
Author: Kip Chemweno
Course: ECE 4424 - Machine Learning
"""

import numpy as np
import pandas as pd
import cv2
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class DrivingDataset(Dataset):
    """
    Dataset class for loading driving images and steering angles.
    Supports both real-world and CARLA-generated data.
    """
    
    def __init__(self, 
                 data_dir: str, 
                 csv_file: str,
                 transform=None,
                 augment: bool = False,
                 data_source: str = 'real'):
        """
        Args:
            data_dir: Root directory containing images
            csv_file: Path to CSV file with image paths and steering angles
            transform: Optional transform to be applied on images
            augment: Whether to apply data augmentation
            data_source: 'real', 'carla', or 'hybrid'
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.augment = augment
        self.data_source = data_source
        
        # Load CSV with image paths and steering angles
        self.data_frame = pd.read_csv(csv_file)
        
        # Ensure required columns exist
        required_cols = ['image_path', 'steering_angle']
        if not all(col in self.data_frame.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        print(f"Loaded {len(self.data_frame)} samples from {data_source} data")
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path and steering angle
        img_name = os.path.join(self.data_dir, self.data_frame.iloc[idx]['image_path'])
        steering_angle = self.data_frame.iloc[idx]['steering_angle']
        
        # Load image
        image = cv2.imread(img_name)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_name}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation if enabled
        if self.augment:
            image, steering_angle = self._augment(image, steering_angle)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(steering_angle, dtype=torch.float32)
    
    def _augment(self, image: np.ndarray, steering: float) -> Tuple[np.ndarray, float]:
        """
        Apply data augmentation techniques.
        """
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            steering = -steering
        
        # Random brightness adjustment
        if np.random.rand() > 0.5:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            brightness = 0.25 + np.random.uniform()
            hsv[:,:,2] = hsv[:,:,2] * brightness
            hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Random shadow
        if np.random.rand() > 0.5:
            image = self._add_random_shadow(image)
        
        return image, steering
    
    def _add_random_shadow(self, image: np.ndarray) -> np.ndarray:
        """Add random shadow to image."""
        h, w = image.shape[:2]
        x1, y1 = w * np.random.rand(), 0
        x2, y2 = w * np.random.rand(), h
        
        mask = np.zeros_like(image[:,:,0])
        mask = cv2.fillPoly(mask, np.array([[(x1,y1), (x2,y2), (w,h), (w,0)]], dtype=np.int32), 255)
        
        shadow_intensity = 0.5
        image = image.copy()
        image[mask == 255] = image[mask == 255] * shadow_intensity
        
        return image.astype(np.uint8)


class ImagePreprocessor:
    """Preprocessing utilities for driving images."""
    
    @staticmethod
    def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (66, 200)) -> np.ndarray:
        """
        Preprocess image following PilotNet specifications.
        
        Args:
            image: Input RGB image
            target_size: Target dimensions (height, width)
        
        Returns:
            Preprocessed image normalized to [-1, 1]
        """
        # Resize
        processed = cv2.resize(image, (target_size[1], target_size[0]))
        
        # Normalize to [-1, 1]
        processed = processed.astype(np.float32) / 127.5 - 1.0
        
        return processed
    
    @staticmethod
    def crop_roi(image: np.ndarray, top_crop: float = 0.35, bottom_crop: float = 0.1) -> np.ndarray:
        """
        Crop region of interest (remove sky and hood).
        
        Args:
            image: Input image
            top_crop: Fraction to crop from top
            bottom_crop: Fraction to crop from bottom
        """
        h = image.shape[0]
        top = int(h * top_crop)
        bottom = int(h * (1 - bottom_crop))
        return image[top:bottom, :, :]


def create_dataloaders(real_data_dir: str,
                       real_csv: str,
                       carla_data_dir: Optional[str] = None,
                       carla_csv: Optional[str] = None,
                       batch_size: int = 32,
                       val_split: float = 0.2,
                       num_workers: int = 4,
                       training_mode: str = 'real',
                       real_val_csv: Optional[str] = None,
                       carla_val_csv: Optional[str] = None) -> dict:
    """
    Create dataloaders for different training scenarios.

    Args:
        real_data_dir: Directory with real images
        real_csv: Training CSV for real data
        real_val_csv: Optional separate validation CSV for real data
        training_mode: 'real', 'carla', or 'hybrid'
        val_split: Validation split ratio (used only if no val_csv provided)

    Returns:
        Dictionary with train and validation dataloaders
    """
    from torchvision import transforms

    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((66, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Check if we have separate validation CSVs
    has_separate_val = (real_val_csv is not None) or (carla_val_csv is not None)

    if has_separate_val:
        # Use provided train/val split
        train_datasets = []
        val_datasets = []

        if training_mode in ['real', 'hybrid']:
            train_datasets.append(DrivingDataset(
                real_data_dir,
                real_csv,
                transform=transform,
                augment=True,
                data_source='real'
            ))

            if real_val_csv:
                val_datasets.append(DrivingDataset(
                    real_data_dir,
                    real_val_csv,
                    transform=transform,
                    augment=False,  # No augmentation on validation
                    data_source='real'
                ))

        if training_mode in ['carla', 'hybrid']:
            if carla_data_dir is None or carla_csv is None:
                raise ValueError("CARLA data paths required for 'carla' or 'hybrid' mode")

            train_datasets.append(DrivingDataset(
                carla_data_dir,
                carla_csv,
                transform=transform,
                augment=True,
                data_source='carla'
            ))

            if carla_val_csv:
                val_datasets.append(DrivingDataset(
                    carla_data_dir,
                    carla_val_csv,
                    transform=transform,
                    augment=False,
                    data_source='carla'
                ))

        # Combine datasets
        if len(train_datasets) > 1:
            train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        else:
            train_dataset = train_datasets[0]

        if len(val_datasets) > 1:
            val_dataset = torch.utils.data.ConcatDataset(val_datasets)
        elif len(val_datasets) == 1:
            val_dataset = val_datasets[0]
        else:
            # No val CSV provided, split from train
            train_size = int((1 - val_split) * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

    else:
        # Original behavior: combine all data then split
        datasets_list = []

        if training_mode in ['real', 'hybrid']:
            real_dataset = DrivingDataset(
                real_data_dir,
                real_csv,
                transform=transform,
                augment=True,
                data_source='real'
            )
            datasets_list.append(real_dataset)

        if training_mode in ['carla', 'hybrid']:
            if carla_data_dir is None or carla_csv is None:
                raise ValueError("CARLA data paths required for 'carla' or 'hybrid' mode")

            carla_dataset = DrivingDataset(
                carla_data_dir,
                carla_csv,
                transform=transform,
                augment=True,
                data_source='carla'
            )
            datasets_list.append(carla_dataset)

        # Combine datasets if hybrid
        if len(datasets_list) > 1:
            full_dataset = torch.utils.data.ConcatDataset(datasets_list)
        else:
            full_dataset = datasets_list[0]

        # Split into train and validation
        train_size = int((1 - val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    return {
        'train': train_loader,
        'val': val_loader
    }


if __name__ == "__main__":
    # Example usage
    print("Data pipeline module loaded successfully")