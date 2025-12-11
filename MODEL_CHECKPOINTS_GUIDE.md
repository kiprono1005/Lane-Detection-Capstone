# Model Checkpoints Guide

This document explains how to obtain and use the trained model checkpoints for this project.

## 📦 Available Models

Three trained models from our comparative study:

| Model Name | Description | Test Accuracy (±3°) | MAE | File Size |
|------------|-------------|---------------------|-----|-----------|
| **pilotnet_tusimple** | Trained on real-world TuSimple data | 72.4% | 5.02° | ~1MB |
| **pilotnet_carla** | Trained on CARLA synthetic data | 59.1% | 2.90° | ~1MB |
| **pilotnet_hybrid** | Trained on 50-50 hybrid dataset | **90.0%** | **1.43°** | ~1MB |

## 🔗 Downloading Model Checkpoints

### Option 1: From Google Drive

If the models are too large for GitHub:

1. Visit: [Google Drive Link](https://drive.google.com/drive/folders/1agAfNJEFfjcPRiJlJMiiTSA2wJacuq7l?usp=sharing)
2. Download `checkpoints.zip`
3. Extract to `./checkpoints/` directory

### Option 2: Train Your Own

Follow the instructions in the main README to train models from scratch:

```bash
# Train real model (~18 minutes)
python train.py --data_source tusimple --epochs 30

# Train CARLA model (~22 minutes)
python train.py --data_source carla --epochs 30

# Train hybrid model (~17 minutes)
python create_hybrid_dataset.py --mode equal
python train.py --data_source hybrid --epochs 30
```

## 📂 Expected Directory Structure

After downloading, your directory should look like:

```
checkpoints/
├── pilotnet_tusimple_20251207_200837/
│   ├── best_model.pth
│   ├── final_model.pth
│   ├── training_history.json
│   └── results.txt
│
├── pilotnet_carla_20251207_163848/
│   ├── best_model.pth
│   ├── final_model.pth
│   ├── training_history.json
│   └── results.txt
│
└── pilotnet_hybrid_20251207_211857/
    ├── best_model.pth
    ├── final_model.pth
    ├── training_history.json
    └── results.txt
```

## 🔍 Checkpoint Contents

Each checkpoint file (`best_model.pth`) contains:

```python
{
    'epoch': int,                    # Training epoch when saved
    'model_state_dict': OrderedDict, # Model weights
    'optimizer_state_dict': dict,    # Optimizer state
    'scheduler_state_dict': dict,    # LR scheduler state
    'best_val_loss': float,          # Best validation loss achieved
    'metrics': dict,                 # Performance metrics
    'history': dict                  # Training history
}
```

## 💻 Loading Models

### Basic Loading

```python
import torch
from model import PilotNet

# Initialize model
model = PilotNet()

# Load checkpoint
checkpoint_path = './checkpoints/pilotnet_hybrid_20251207_211857/best_model.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
```

### Loading with Evaluation

```python
from evaluation import ModelEvaluator
from data_pipeline import create_dataloaders

# Load model
model = PilotNet()
checkpoint = torch.load('path/to/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Load test data
dataloaders = create_dataloaders(
    real_data_dir='./data/processed/tusimple_images',
    real_csv='./data/processed/tusimple_train_steering.csv',
    real_val_csv='./data/processed/tusimple_val_steering.csv',
    batch_size=32,
    num_workers=0,
    training_mode='real'
)

# Evaluate
evaluator = ModelEvaluator(model, device='cuda')
metrics, predictions, targets = evaluator.evaluate_on_loader(dataloaders['val'])

print(f"MAE: {metrics['mae']:.4f} rad ({metrics['mae']*57.3:.2f}°)")
print(f"Accuracy ±3°: {metrics['accuracy_5deg']:.1f}%")
```

### Inference on Single Image

```python
import torch
import cv2
from torchvision import transforms
from model import PilotNet

# Load model
model = PilotNet()
checkpoint = torch.load('path/to/best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare image transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load and preprocess image
image = cv2.imread('path/to/image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Predict
with torch.no_grad():
    steering_angle = model(image_tensor).item()

print(f"Predicted steering: {steering_angle:.4f} rad ({steering_angle*57.3:.2f}°)")
```

## 📊 Checkpoint Metadata

### TuSimple Model (Real-World)

```json
{
    "model": "PilotNet",
    "dataset": "TuSimple",
    "training_samples": 2901,
    "validation_samples": 725,
    "epochs_trained": 18,
    "best_epoch": 16,
    "best_val_loss": 0.0512,
    "best_val_mae": 0.0876,
    "accuracy_3deg": 72.4,
    "accuracy_6deg": 89.1,
    "r2_score": 0.7234,
    "training_time_minutes": 18
}
```

### CARLA Model (Synthetic)

```json
{
    "model": "PilotNet",
    "dataset": "CARLA",
    "training_samples": 2400,
    "validation_samples": 600,
    "epochs_trained": 22,
    "best_epoch": 19,
    "best_val_loss": 0.0982,
    "best_val_mae": 0.0506,
    "accuracy_3deg": 59.1,
    "accuracy_6deg": 88.4,
    "r2_score": 0.5891,
    "training_time_minutes": 23
}
```

### Hybrid Model (Best Performance)

```json
{
    "model": "PilotNet",
    "dataset": "Hybrid (50% TuSimple + 50% CARLA)",
    "training_samples": 2400,
    "validation_samples": 600,
    "epochs_trained": 16,
    "best_epoch": 14,
    "best_val_loss": 0.0231,
    "best_val_mae": 0.0250,
    "accuracy_3deg": 90.0,
    "accuracy_6deg": 97.8,
    "r2_score": 0.9000,
    "training_time_minutes": 17
}
```

## ⚠️ Important Notes

1. **Evaluation Dataset:** All models were evaluated on the same TuSimple validation set (725 samples) for fair comparison.

2. **Input Format:** Models expect:
   - RGB images resized to 66×200 pixels
   - Normalized to [-1, 1] range
   - Batch dimension: (N, 3, 66, 200)

3. **Output Format:** 
   - Single steering angle value
   - Range: [-0.436, 0.436] radians (±25°)
   - Negative = left turn, Positive = right turn

4. **Hardware Requirements:**
   - Loading models: Any CPU
   - Inference: CPU or GPU (GPU recommended for batch processing)

---

**Last Updated:** December 10, 2025  
**Checkpoint Version:** v1.0
