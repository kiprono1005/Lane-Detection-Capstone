# Model Checkpoints Guide

How to obtain, load and evaluate the trained models for this project.

> **Updated November 2026.** The accuracy figures in the original version of this guide (hybrid 90.0%, real 72.4%, CARLA 59.1%) came from an evaluation that mixed models and test sets and did not hold up. Every number below comes from `eval_clean.py` on **TuSimple's official test set (2,782 frames no model trained on)**, with augmentation off. See the README for the full correction.

## 📦 Checkpoints

| Checkpoint folder | Training data | Test MAE | Within ±3° | In Google Drive zip |
|---|---|---|---|---|
| `pilotnet_tusimple_20251207_200837` | Real only (seed 42) | 1.23° | 92.9% | ✅ |
| `pilotnet_tusimple_s42` | Real only (seed 42, retrained) | 1.23° | 92.9% | — |
| `pilotnet_tusimple_s1` | Real only (seed 1) | 1.15° | 93.7% | — |
| `pilotnet_hybrid_augment` | All real + all CARLA v2 | 1.30° | 91.8% | — |
| `pilotnet_hybrid_v2` | 50/50 real + CARLA v2 | 1.66° | 84.5% | — |
| `pilotnet_hybrid_20251207_211857` | 50/50 real + CARLA v1 | 1.75° | 83.2% | ✅ |
| `pilotnet_carla_v2` | CARLA v2 only | 2.22° | 70.7% | — |
| `pilotnet_carla_20251207_163848` | CARLA v1 only | 3.29° | 54.9% | ✅ |

For scale, always predicting the training-mean angle scores 1.94° (78.3% within ±3°), and always predicting straight scores 2.23° (70.7%).

**CARLA v1** is the original synthetic dataset. **v2** was re-collected after fixing a color-channel bug in `carla_data_collection.py` (red and blue were swapped when saving frames).

"Within ±3°" means |error| < 0.05 rad (2.86°).

## 🔗 Getting Checkpoints

### Option 1: Google Drive (December 2025 models)

1. Visit the [Google Drive folder](https://drive.google.com/drive/folders/1agAfNJEFfjcPRiJlJMiiTSA2wJacuq7l?usp=sharing).
2. Download `checkpoints.zip`.
3. Extract to `./checkpoints/`.

The zip contains the three December 2025 checkpoints marked ✅ above. The November 2026 re-runs are not in it; train them with Option 2.

### Option 2: Train Your Own

```bash
# Real only
python train.py --data_source tusimple --num_workers 0 --seed 42 --experiment_name pilotnet_tusimple_s42
python train.py --data_source tusimple --num_workers 0 --seed 1  --experiment_name pilotnet_tusimple_s1

# CARLA only (needs ./data/carla)
python train.py --data_source carla --num_workers 0 --experiment_name pilotnet_carla_v2

# 50/50 mix
python create_hybrid_dataset.py --mode equal --total_samples 3000
python train.py --data_source hybrid --num_workers 0 --experiment_name pilotnet_hybrid_v2

# All real + all CARLA (overwrites ./data/hybrid; rename the 50/50 folder first to keep it)
python create_hybrid_dataset.py --mode augment
python train.py --data_source hybrid --num_workers 0 --experiment_name pilotnet_hybrid_augment
```

Training seed 42 reproduces the December real-only checkpoint almost exactly (`train.py` sets deterministic cuDNN). Expect roughly 15–35 minutes per model on an RTX 5070.

## 📂 Folder Contents

```
checkpoints/
└── pilotnet_<source>_<name>/
    ├── best_model.pth          # weights with the lowest validation loss  ← use this
    ├── final_model.pth         # weights at the last epoch
    ├── checkpoint_epoch_N.pth  # every 5 epochs
    ├── training_history.json   # per-epoch loss, MAE, learning rate
    └── results.txt             # validation metrics written at the end of training
```

`results.txt` reports each model on **its own validation split**, which differs between models (CARLA-only is validated on CARLA frames). Don't compare models using those numbers; use `eval_clean.py`.

## 🔍 Checkpoint Contents

Each `.pth` file is a dictionary:

```python
{
    'epoch': int,                 # 0-based epoch when saved
    'model_state_dict': ...,      # model weights
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'best_val_loss': float,
    'metrics': dict,              # validation metrics, incl. NumPy prediction arrays
    'history': dict               # training history up to that epoch
}
```

## 💻 Loading a Model

Because `metrics` contains NumPy arrays, PyTorch 2.6+ needs `weights_only=False`. Only do this for checkpoints you trust.

```python
import torch
from model import PilotNet

model = PilotNet()
ckpt = torch.load('./checkpoints/pilotnet_tusimple_s1/best_model.pth',
                  map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Epoch {ckpt['epoch']}, best validation loss {ckpt['best_val_loss']:.4f}")
```

## 📏 Evaluating Checkpoints

Use `eval_clean.py`. It labels TuSimple's official test set with the same function used for training, scores up to three models with augmentation off, and prints MAE, RMSE, R², threshold accuracy, two baselines, error by steering direction, and a 95% interval on the real-versus-hybrid gap.

```bash
python eval_clean.py \
  --ckpt_real   checkpoints/pilotnet_tusimple_s1/best_model.pth \
  --ckpt_carla  checkpoints/pilotnet_carla_v2/best_model.pth \
  --ckpt_hybrid checkpoints/pilotnet_hybrid_augment/best_model.pth \
  --out results/clean_eval_example
```

The hybrid row is labeled "Hybrid 50/50" in the output regardless of which hybrid checkpoint you pass.

## 🖼️ Inference on a Single Image

```python
import torch, cv2
from torchvision import transforms
from model import PilotNet

model = PilotNet()
ckpt = torch.load('path/to/best_model.pth', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

image = cv2.cvtColor(cv2.imread('path/to/image.jpg'), cv2.COLOR_BGR2RGB)
with torch.no_grad():
    angle = model(transform(image).unsqueeze(0)).item()
print(f"Predicted steering: {angle:.4f} rad ({angle * 57.2958:.2f}°)")
```

## ⚠️ Notes

1. **Input:** RGB, resized to 66×200, normalized to [−1, 1], shape (N, 3, 66, 200).
2. **Output:** one value in radians, clipped in training to ±0.436 (±25°). Negative is left, following the convention in `process_tusimple.py`.
3. **What the label means:** for real data it's the lane-marking center's offset from the image center, not a measured steering angle. For CARLA data it's the simulated driver's steering command. See the README for why that difference matters.

---

**Last updated:** November 2026
