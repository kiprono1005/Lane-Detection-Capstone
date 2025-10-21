"""
Main Training Script for Autonomous Lane Keeping
Run this script to train models on processed lane detection data

Usage:
    # Process data first (if not done)
    python process_tusimple.py --path ./data/kaggle

    # Then train
    python train.py --data_source tusimple --epochs 30
    python train.py --data_source roboflow --epochs 30
    python train.py --data_source kaggle --epochs 30

Author: Kip Chemweno
Course: ECE 4424 - Machine Learning
"""

import argparse
import torch
import os
from pathlib import Path

# Import project modules
from data_pipeline import create_dataloaders
from model import PilotNet, SimplifiedPilotNet
from training import Trainer
from evaluation import ModelEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train lane keeping model on processed data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument('--data_source', type=str, default='tusimple',
                       choices=['tusimple', 'roboflow', 'carla', 'hybrid'],
                       help='Which dataset to use for training')
    parser.add_argument('--processed_dir', type=str, default='./data/processed',
                       help='Directory containing processed data')

    # Model arguments
    parser.add_argument('--model', type=str, default='pilotnet',
                       choices=['pilotnet', 'simplified'],
                       help='Model architecture')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for optimizer')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio (if no separate val set)')
    parser.add_argument('--early_stopping', type=int, default=10,
                       help='Early stopping patience')

    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to train on')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for tensorboard logs')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_paths(processed_dir: str, data_source: str):
    """Get paths to processed data based on source."""
    processed_dir = Path(processed_dir)

    if data_source == 'kaggle':
        return {
            'images_dir': str(processed_dir / 'kaggle_images'),
            'train_csv': str(processed_dir / 'kaggle_train_steering.csv'),
            'val_csv': str(processed_dir / 'kaggle_val_steering.csv')
        }
    elif data_source == 'roboflow':
        return {
            'images_dir': str(processed_dir / 'roboflow_images'),
            'train_csv': str(processed_dir / 'roboflow_train_steering.csv'),
            'val_csv': str(processed_dir / 'roboflow_valid_steering.csv')
        }
    elif data_source == 'tusimple':
        return {
            'images_dir': str(processed_dir / 'tusimple_images'),
            'train_csv': str(processed_dir / 'tusimple_train_steering.csv'),
            'val_csv': str(processed_dir / 'tusimple_val_steering.csv')
        }
    elif data_source == 'both':
        # For 'both', we'll need to modify data_pipeline to handle multiple sources
        return {
            'kaggle_images': str(processed_dir / 'kaggle_images'),
            'kaggle_train_csv': str(processed_dir / 'kaggle_train_steering.csv'),
            'kaggle_val_csv': str(processed_dir / 'kaggle_val_steering.csv'),
            'roboflow_images': str(processed_dir / 'roboflow_images'),
            'roboflow_train_csv': str(processed_dir / 'roboflow_train_steering.csv'),
            'roboflow_val_csv': str(processed_dir / 'roboflow_valid_steering.csv')
        }


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Auto-generate experiment name if not provided
    if args.experiment_name is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.model}_{args.data_source}_{timestamp}"

    print(f"\n{'='*80}")
    print(f"AUTONOMOUS LANE KEEPING TRAINING")
    print(f"{'='*80}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Data source: {args.data_source}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"{'='*80}\n")

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'

    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

    # Get data paths
    data_paths = get_data_paths(args.processed_dir, args.data_source)

    # Check if processed data exists
    print("Checking for processed data...")
    if args.data_source in ['kaggle', 'roboflow', 'tusimple']:
        train_csv_path = Path(data_paths['train_csv'])
        images_dir_path = Path(data_paths['images_dir'])

        if not train_csv_path.exists():
            print(f"\n✗ ERROR: Processed data not found!")
            print(f"  Missing: {train_csv_path}")
            print(f"\nYou need to process the data first:")

            if args.data_source == 'tusimple':
                print(f"  python process_tusimple.py --path ./data/kaggle")
            elif args.data_source == 'kaggle':
                print(f"  python lane_to_steering.py --dataset kaggle --path ./data/kaggle --splits train val")
            elif args.data_source == 'roboflow':
                print(f"  python lane_to_steering.py --dataset roboflow --path ./data/roboflow --splits train valid")

            return

        if not images_dir_path.exists():
            print(f"\n✗ ERROR: Images directory not found!")
            print(f"  Missing: {images_dir_path}")
            return

        # Count images
        num_images = len(list(images_dir_path.glob('*.jpg'))) + len(list(images_dir_path.glob('*.png')))
        print(f"✓ Found {num_images} images in {images_dir_path}")
        print(f"✓ Found train CSV: {train_csv_path}")

    print("✓ Processed data found\n")

    # Create data loaders
    print("Loading datasets...")
    try:
        if args.data_source == 'both':
            print("⚠️  'both' mode not yet implemented in data_pipeline.py")
            print("   Please use --data_source tusimple, kaggle, or roboflow")
            return
        else:
            dataloaders = create_dataloaders(
                real_data_dir=data_paths['images_dir'],
                real_csv=data_paths['train_csv'],
                batch_size=args.batch_size,
                val_split=args.val_split,
                num_workers=args.num_workers,
                training_mode='real'  # Using 'real' mode for all processed data
            )
        print("✓ Datasets loaded successfully\n")
    except Exception as e:
        print(f"✗ Error loading datasets: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create model
    print("Initializing model...")
    if args.model == 'pilotnet':
        model = PilotNet(dropout_rate=args.dropout)
    else:
        model = SimplifiedPilotNet(dropout_rate=args.dropout)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {num_params:,} parameters\n")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        device=args.device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name
    )

    # Train
    print("Starting training...\n")
    try:
        trainer.train(
            num_epochs=args.epochs,
            early_stopping_patience=args.early_stopping
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current model state...")
        trainer.save_checkpoint('interrupted_model.pth', {})
        print("✓ Model saved")
    except Exception as e:
        print(f"\n✗ Training error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Evaluate best model
    print("\n" + "="*80)
    print("EVALUATING BEST MODEL ON VALIDATION SET")
    print("="*80 + "\n")

    # Load best checkpoint
    best_checkpoint = trainer.checkpoint_dir / 'best_model.pth'
    if best_checkpoint.exists():
        print(f"Loading best model from {best_checkpoint}")
        trainer.load_checkpoint(str(best_checkpoint))

        # Evaluate
        evaluator = ModelEvaluator(model, args.device)
        metrics, predictions, targets = evaluator.evaluate_on_loader(dataloaders['val'])
        evaluator.print_metrics(metrics, "Validation")

        # Create visualizations
        results_dir = Path('./results') / args.experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)

        evaluator.plot_predictions(
            predictions,
            targets,
            save_path=str(results_dir / 'predictions.png'),
            dataset_name='Validation'
        )

        evaluator.analyze_error_by_steering_range(predictions, targets)

        print(f"\n✓ Results saved to {results_dir}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nExperiment: {args.experiment_name}")
    print(f"Checkpoints: {trainer.checkpoint_dir}")
    print(f"Logs: {args.log_dir}/{args.experiment_name}")
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir={args.log_dir}/{args.experiment_name}")
    print("\n")


if __name__ == "__main__":
    main()