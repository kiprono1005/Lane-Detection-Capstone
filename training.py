"""
Training Pipeline for Autonomous Lane Keeping

Author: Kip Chemweno
Course: ECE 4424 - Machine Learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from pathlib import Path
import json
from typing import Dict
from tqdm import tqdm


class Trainer:
    """Training manager for lane keeping models."""

    def __init__(self, model, train_loader, val_loader, device='cuda',
                 learning_rate=1e-4, weight_decay=1e-5,
                 checkpoint_dir='./checkpoints', log_dir='./logs',
                 experiment_name='pilotnet_real'):

        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Learning rate scheduler (FIXED - removed verbose parameter)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Directories
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{experiment_name}")

        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_mae': [], 'val_mae': [],
            'learning_rates': []
        }

        self.best_val_loss = float('inf')
        self.epoch = 0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss, running_mae = 0.0, 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} [Train]")

        for batch_idx, (images, steering_angles) in enumerate(pbar):
            images = images.to(self.device)
            steering_angles = steering_angles.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.criterion(predictions, steering_angles)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            mae = torch.mean(torch.abs(predictions - steering_angles))
            running_loss += loss.item()
            running_mae += mae.item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mae': f'{mae.item():.4f}'})

            global_step = self.epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)

        return {'loss': running_loss / num_batches, 'mae': running_mae / num_batches}

    def validate(self) -> Dict:
        """Validate the model."""
        self.model.eval()
        running_loss, running_mae = 0.0, 0.0
        all_predictions, all_targets = [], []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch+1} [Val]")

            for images, steering_angles in pbar:
                images = images.to(self.device)
                steering_angles = steering_angles.to(self.device).unsqueeze(1)

                predictions = self.model(images)
                loss = self.criterion(predictions, steering_angles)
                mae = torch.mean(torch.abs(predictions - steering_angles))

                running_loss += loss.item()
                running_mae += mae.item()

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(steering_angles.cpu().numpy())

                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mae': f'{mae.item():.4f}'})

        num_batches = len(self.val_loader)
        return {
            'loss': running_loss / num_batches,
            'mae': running_mae / num_batches,
            'predictions': np.array(all_predictions),
            'targets': np.array(all_targets)
        }

    def train(self, num_epochs: int, early_stopping_patience: int = 10):
        """Complete training loop."""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        epochs_without_improvement = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            self.epoch = epoch

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']

            # Store history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['learning_rates'].append(current_lr)

            # Log to tensorboard
            self.writer.add_scalar('Epoch/TrainLoss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Epoch/ValLoss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Epoch/TrainMAE', train_metrics['mae'], epoch)
            self.writer.add_scalar('Epoch/ValMAE', val_metrics['mae'], epoch)
            self.writer.add_scalar('Epoch/LearningRate', current_lr, epoch)

            # Print summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f} | Train MAE: {train_metrics['mae']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Val MAE: {val_metrics['mae']:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")

            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                print(f"✓ Validation improved: {self.best_val_loss:.4f} → {val_metrics['loss']:.4f}")
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model.pth', val_metrics)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f"✗ No improvement for {epochs_without_improvement} epoch(s)")

            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

            # Periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', val_metrics)

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time/60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")

        self.save_checkpoint('final_model.pth', val_metrics)
        self.save_history()
        self.writer.close()

    def save_checkpoint(self, filename: str, metrics: Dict):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'history': self.history
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def save_history(self):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Saved training history: {history_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        print(f"Loaded checkpoint from epoch {self.epoch}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


if __name__ == "__main__":
    print("Training pipeline module loaded successfully")