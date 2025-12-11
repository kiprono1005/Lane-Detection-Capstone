"""
Main Training Script for Autonomous Lane Keeping
Run this script to train models on processed lane detection data.

Author: Kip Chemweno
Course: ECE 4424 - Machine Learning
"""

import argparse
import os
import random
from pathlib import Path
import datetime

import numpy as np
import torch

from data_pipeline import create_dataloaders
from model import PilotNet, SimplifiedPilotNet
from training import Trainer
from evaluation import ModelEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train lane-keeping model on processed data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data_source",
        type=str,
        default="tusimple",
        choices=["tusimple", "kaggle", "roboflow", "carla", "hybrid"],
        help="Which dataset to use for training",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="./data/processed",
        help="Directory containing processed real datasets (tusimple/kaggle/roboflow)",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="pilotnet",
        choices=["pilotnet", "simplified"],
        help="Model architecture",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate for fully connected layers",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split ratio (used only if no separate val CSV is provided)",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=10,
        help="Early stopping patience (epochs without improvement)",
    )

    # System / logging arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Base directory for saving model checkpoints",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Base directory for TensorBoard logs",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name for this training experiment (auto-generated if not set)",
    )

    return parser.parse_args()


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_paths(processed_dir: str, data_source: str):
    """
    Return a simple dict with the correct image directory and CSV paths
    for the chosen data source.
    """
    processed_dir = Path(processed_dir)

    if data_source == "tusimple":
        return {
            "images_dir": str(processed_dir / "tusimple_images"),
            "train_csv": str(processed_dir / "tusimple_train_steering.csv"),
            "val_csv": str(processed_dir / "tusimple_val_steering.csv"),
        }
    elif data_source == "kaggle":
        # Assumes you've processed Kaggle TuSimple into the same format
        return {
            "images_dir": str(processed_dir / "kaggle_images"),
            "train_csv": str(processed_dir / "kaggle_train_steering.csv"),
            "val_csv": str(processed_dir / "kaggle_val_steering.csv"),
        }
    elif data_source == "roboflow":
        return {
            "images_dir": str(processed_dir / "roboflow_images"),
            "train_csv": str(processed_dir / "roboflow_train_steering.csv"),
            "val_csv": str(processed_dir / "roboflow_valid_steering.csv"),
        }
    elif data_source == "carla":
        carla_dir = Path("./data/carla")
        return {
            "images_dir": str(carla_dir / "images"),
            "train_csv": str(carla_dir / "carla_steering.csv"),
            "val_csv": None,  # will be created via random split
        }
    elif data_source == "hybrid":
        hybrid_dir = Path("./data/hybrid")
        return {
            "images_dir": str(hybrid_dir / "hybrid_images"),
            "train_csv": str(hybrid_dir / "hybrid_train_steering.csv"),
            "val_csv": str(hybrid_dir / "hybrid_val_steering.csv"),
        }
    else:
        raise ValueError(f"Unknown data source: {data_source}")


def main():
    args = parse_args()
    set_seed(args.seed)

    # Experiment name
    if args.experiment_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.model}_{args.data_source}_{timestamp}"

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print("AUTONOMOUS LANE KEEPING TRAINING")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Data source: {args.data_source}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print("=" * 80)

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {mem_gb:.2f} GB\n")
    else:
        print("Using CPU\n")

    # Resolve data paths
    data_paths = get_data_paths(args.processed_dir, args.data_source)

    # Basic existence checks for images and train CSV
    print("Checking for processed data...")
    train_csv_path = Path(data_paths["train_csv"])
    images_dir_path = Path(data_paths["images_dir"])

    if not train_csv_path.exists():
        print("\n✗ ERROR: Training CSV not found!")
        print(f"  Expected: {train_csv_path}")
        return

    if not images_dir_path.exists():
        print("\n✗ ERROR: Images directory not found!")
        print(f"  Expected: {images_dir_path}")
        return

    print("✓ Processed data found")

    # Create dataloaders
    print("Loading datasets...")
    try:
        if args.data_source == "carla":
            # Carla-only training
            dataloaders = create_dataloaders(
                real_data_dir=str(images_dir_path),  # unused when training_mode='carla'
                real_csv=str(train_csv_path),
                carla_data_dir=str(images_dir_path),
                carla_csv=str(train_csv_path),
                batch_size=args.batch_size,
                val_split=args.val_split,
                num_workers=args.num_workers,
                training_mode="carla",
            )
        else:
            val_csv = data_paths.get("val_csv", None)
            if val_csv is not None:
                # Use existing train/val CSV split
                dataloaders = create_dataloaders(
                    real_data_dir=str(images_dir_path),
                    real_csv=str(train_csv_path),
                    real_val_csv=str(val_csv),
                    batch_size=args.batch_size,
                    val_split=args.val_split,
                    num_workers=args.num_workers,
                    training_mode="real",
                )
            else:
                # Fall back to random split
                dataloaders = create_dataloaders(
                    real_data_dir=str(images_dir_path),
                    real_csv=str(train_csv_path),
                    batch_size=args.batch_size,
                    val_split=args.val_split,
                    num_workers=args.num_workers,
                    training_mode="real",
                )

        print("✓ Datasets loaded successfully\n")
    except Exception as e:
        print(f"✗ Error loading datasets: {e}")
        import traceback

        traceback.print_exc()
        return

    # Build model
    print("Initializing model...")
    if args.model == "pilotnet":
        model = PilotNet(dropout_rate=args.dropout)
    else:
        model = SimplifiedPilotNet(dropout_rate=args.dropout)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {num_params:,} parameters\n")

    # Create Trainer (match training.Trainer __init__ exactly)
    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        device=str(device),
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
    )

    # Train
    print("Starting training...\n")
    trainer.train(num_epochs=args.epochs, early_stopping_patience=args.early_stopping)

    # After training, load best model if it exists
    exp_dir = Path(args.checkpoint_dir) / args.experiment_name
    best_model_path = exp_dir / "best_model.pth"
    if best_model_path.exists():
        print(f"\nLoading best model from: {best_model_path}")
        trainer.load_checkpoint(str(best_model_path))
    else:
        print("\nWarning: best_model.pth not found, using final model weights.")

    # Final evaluation on validation set
    print("\nEvaluating model on validation set...")
    evaluator = ModelEvaluator(trainer.model, device=str(device))
    metrics, predictions, targets = evaluator.evaluate_on_loader(dataloaders["val"])

    print("\nValidation metrics:")
    print(f"  MAE:          {metrics['mae']:.6f}")
    print(f"  RMSE:         {metrics['rmse']:.6f}")
    print(f"  R^2 Score:    {metrics['r2_score']:.6f}")
    print(f"  MAPE:         {metrics['mape']:.2f}%")
    print(f"  Acc @5deg:    {metrics['accuracy_5deg']:.2f}%")
    print(f"  Acc @10deg:   {metrics['accuracy_10deg']:.2f}%")
    print(f"  N samples:    {metrics['n_samples']}")

    # Save a small summary next to checkpoints
    results_file = exp_dir / "results.txt"
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        f.write("TRAINING RESULTS SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Experiment: {args.experiment_name}\n")
        f.write(f"Data source: {args.data_source}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Device: {device}\n\n")
        f.write(f"Best model path: {best_model_path}\n\n")
        f.write(f"Validation MAE: {metrics['mae']:.6f}\n")
        f.write(f"Validation RMSE: {metrics['rmse']:.6f}\n")
        f.write(f"R^2 Score: {metrics['r2_score']:.6f}\n")
        f.write(f"MAPE: {metrics['mape']:.2f}%\n")
        f.write(f"Accuracy @5deg: {metrics['accuracy_5deg']:.2f}%\n")
        f.write(f"Accuracy @10deg: {metrics['accuracy_10deg']:.2f}%\n")
        f.write(f"N samples: {metrics['n_samples']}\n")

    print(f"\n✓ Results saved to: {results_file}")
    print("\nTRAINING COMPLETE.\n")


if __name__ == "__main__":
    main()
