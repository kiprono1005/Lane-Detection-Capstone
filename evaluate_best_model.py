"""
Evaluate Best Model - Manual Evaluation Script
Run this after training to evaluate your best model

Author: Kip Chemweno
"""

import torch
from pathlib import Path
from model import PilotNet
from evaluation import ModelEvaluator
from data_pipeline import create_dataloaders


def main():
    # Configuration
    CHECKPOINT_PATH = './checkpoints/pilotnet_tusimple_20251020_000538/best_model.pth'
    DATA_DIR = './data/processed/tusimple_images'
    TRAIN_CSV = './data/processed/tusimple_train_steering.csv'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 32

    print("="*80)
    print("EVALUATING BEST MODEL")
    print("="*80)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Device: {DEVICE}")
    print()

    # Load model
    print("Loading model...")
    model = PilotNet()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"✓ Best validation loss: {checkpoint['best_val_loss']:.4f}")
    print()

    # Load data (IMPORTANT: num_workers=0 for Windows)
    print("Loading validation data...")
    dataloaders = create_dataloaders(
        real_data_dir=DATA_DIR,
        real_csv=TRAIN_CSV,
        batch_size=BATCH_SIZE,
        val_split=0.2,
        num_workers=0,  # CHANGED: 0 for Windows compatibility
        training_mode='real'
    )
    print(f"✓ Validation samples: {len(dataloaders['val'].dataset)}")
    print()

    # Evaluate
    print("Evaluating model...")
    evaluator = ModelEvaluator(model, device=DEVICE)
    metrics, predictions, targets = evaluator.evaluate_on_loader(dataloaders['val'])

    # Print metrics
    evaluator.print_metrics(metrics, "Validation")

    # Create visualizations
    results_dir = Path('./results/pilotnet_tusimple_20251020_000538')
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Creating visualizations...")
    evaluator.plot_predictions(
        predictions,
        targets,
        save_path=str(results_dir / 'predictions.png'),
        dataset_name='Validation'
    )

    # Error analysis by steering range
    evaluator.analyze_error_by_steering_range(predictions, targets)

    print(f"\n✓ Results saved to {results_dir}")
    print(f"✓ Predictions plot: {results_dir / 'predictions.png'}")
    print()

    print("="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print("\nKey Metrics Summary:")
    print(f"  MAE: {metrics['mae']:.4f} rad ({metrics['mae']*57.3:.2f}°)")
    print(f"  RMSE: {metrics['rmse']:.4f} rad")
    print(f"  R² Score: {metrics['r2_score']:.4f}")
    print(f"  Accuracy (±3°): {metrics['accuracy_5deg']:.1f}%")
    print(f"  Accuracy (±6°): {metrics['accuracy_10deg']:.1f}%")
    print()


if __name__ == '__main__':
    main()