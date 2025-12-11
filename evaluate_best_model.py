"""
Evaluate Best Model - Flexible Version
Works with TuSimple, CARLA, and Hybrid models

Usage:
    python evaluate_model.py --model_type tusimple
    python evaluate_model.py --model_type carla
    python evaluate_model.py --model_type hybrid

Author: Kip Chemweno
"""

import torch
import argparse
from pathlib import Path
from model import PilotNet
from evaluation import ModelEvaluator
from data_pipeline import create_dataloaders


def get_model_config(model_type):
    """Get configuration for each model type."""

    if model_type == 'tusimple':
        return {
            'checkpoint': './checkpoints/pilotnet_tusimple_20251207_200837/best_model.pth',
            'data_dir': './data/processed/tusimple_images',
            'train_csv': './data/processed/tusimple_train_steering.csv',
            'val_csv': './data/processed/tusimple_val_steering.csv',
            'results_dir': './results/pilotnet_tusimple_20251207_200837',
            'name': 'TuSimple (Real)'
        }

    elif model_type == 'carla':
        return {
            'checkpoint': './checkpoints/pilotnet_carla_20251207_163848/best_model.pth',
            'data_dir': './data/carla/images',
            'train_csv': './data/carla/carla_steering.csv',
            'val_csv': None,  # CARLA doesn't have separate val CSV
            'results_dir': './results/pilotnet_carla_20251207_163848',
            'name': 'CARLA (Synthetic)'
        }

    elif model_type == 'hybrid':
        return {
            'checkpoint': './checkpoints/pilotnet_hybrid_20251207_211857/best_model.pth',
            'data_dir': './data/hybrid/hybrid_images',
            'train_csv': './data/hybrid/hybrid_train_steering.csv',
            'val_csv': './data/hybrid/hybrid_val_steering.csv',
            'results_dir': './results/pilotnet_hybrid_20251207_211857',
            'name': 'Hybrid (Real + CARLA)'
        }

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['tusimple', 'carla', 'hybrid'],
                       help='Which model to evaluate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')

    args = parser.parse_args()

    # Get configuration
    config = get_model_config(args.model_type)

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("="*80)
    print(f"EVALUATING {config['name'].upper()} MODEL")
    print("="*80)
    print(f"Checkpoint: {config['checkpoint']}")
    print(f"Device: {device}")
    print()

    # Check if checkpoint exists
    checkpoint_path = Path(config['checkpoint'])
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print(f"\nAvailable checkpoints in ./checkpoints/:")
        checkpoints_dir = Path('./checkpoints')
        if checkpoints_dir.exists():
            for ckpt_dir in sorted(checkpoints_dir.iterdir()):
                if ckpt_dir.is_dir():
                    best_model = ckpt_dir / 'best_model.pth'
                    if best_model.exists():
                        print(f"  {ckpt_dir.name}/best_model.pth")
        return

    # Load model
    print("Loading model...")
    model = PilotNet()
    checkpoint = torch.load(config['checkpoint'], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"✓ Best validation loss: {checkpoint['best_val_loss']:.4f}")
    print()

    # Load data
    print("Loading validation data...")
    dataloaders = create_dataloaders(
        real_data_dir=config['data_dir'],
        real_csv=config['train_csv'],
        real_val_csv=config['val_csv'],  # None for CARLA (will auto-split)
        batch_size=args.batch_size,
        val_split=0.2,
        num_workers=0,  # Windows compatibility
        training_mode='real'
    )

    print(f"✓ Validation samples: {len(dataloaders['val'].dataset)}")
    print()

    # Evaluate
    print("Evaluating model...")
    evaluator = ModelEvaluator(model, device=device)
    metrics, predictions, targets = evaluator.evaluate_on_loader(dataloaders['val'])

    # Print metrics
    evaluator.print_metrics(metrics, f"{config['name']} Validation")

    # Create visualizations
    results_dir = Path(config['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Creating visualizations...")
    evaluator.plot_predictions(
        predictions,
        targets,
        save_path=str(results_dir / 'predictions.png'),
        dataset_name=config['name']
    )

    # Error analysis
    evaluator.analyze_error_by_steering_range(predictions, targets)

    print(f"\n✓ Results saved to {results_dir}")
    print(f"✓ Predictions plot: {results_dir / 'predictions.png'}")
    print()

    print("="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\n{config['name']} Summary:")
    print(f"  MAE: {metrics['mae']:.4f} rad ({metrics['mae']*57.3:.2f}°)")
    print(f"  RMSE: {metrics['rmse']:.4f} rad")
    print(f"  R² Score: {metrics['r2_score']:.4f}")
    print(f"  Accuracy (±3°): {metrics['accuracy_5deg']:.1f}%")
    print(f"  Accuracy (±6°): {metrics['accuracy_10deg']:.1f}%")
    print()


if __name__ == '__main__':
    main()