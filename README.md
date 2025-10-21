# Autonomous Lane Keeping Using Machine Learning

**Converting Lane Detection Labels to Steering Predictions**

**Author:** Kip Chemweno  
**Course:** ECE 4424 - Machine Learning  
**Institution:** Virginia Tech  
**Date:** October 2025

---

## 📋 Project Overview

This project implements an end-to-end deep learning system for autonomous lane keeping by:
1. **Converting lane point annotations** from the TuSimple dataset into **steering angles**
2. **Training a CNN** (PilotNet architecture) to predict steering directly from images
3. **Achieving strong baseline performance**: 72.4% accuracy within ±3° (1.13° MAE)

**Key Innovation:** Geometric algorithm that generates steering angle labels from lane waypoint annotations, enabling training on lane detection datasets without direct steering measurements.

---

## 🎯 Research Question

How effectively can we train lane-keeping models using steering angles derived from lane detection datasets, and what is the optimal approach for converting lane annotations to steering predictions?

---

## 📁 Project Structure

```
autonomous-lane-keeping/
├── data/
│   ├── kaggle/
│   │   └── train_set/              # TuSimple dataset from Kaggle
│   │       ├── clips/              # Video frames (.jpg)
│   │       ├── label_data_0313.json
│   │       ├── label_data_0531.json
│   │       └── label_data_0601.json
│   └── processed/
│       ├── tusimple_images/        # Processed training images
│       ├── tusimple_train_steering.csv
│       └── tusimple_val_steering.csv
├── checkpoints/                     # Saved model checkpoints
├── logs/                            # TensorBoard logs
├── results/                         # Evaluation results and plots
├── eda_results/                     # Exploratory data analysis
├── data_pipeline.py                # Data loading and augmentation
├── eda_analysis.py                 # Exploratory data analysis
├── evaluate_best_model.py          # Model evaluation script
├── evaluation.py                   # Metrics and evaluation
├── lane_to_steering.py             # Lane Data to Steering Data
├── model.py                        # PilotNet CNN architecture
├── process_tusimple.py             # 🔑 Main data processor
├── README.md                       # This file
├── requirements.txt                # Dependencies
├── test_setup.py                   # Setup verification
├── train.py                        # Main training script
├── training.py                     # Training loop
└── visualization_utils.py          # Result visualization
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/kiprono1005/Lane-Detection-Capstone.git
cd autonomous-lane-keeping

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For GPU support (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Download TuSimple Dataset

**Option A: Kaggle API (Recommended)**
```bash
pip install kaggle

# Get API credentials from kaggle.com/settings
# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<you>\.kaggle\ (Windows)

kaggle datasets download -d manideep1108/tusimple
unzip tusimple.zip -d ./data/kaggle
```

**Option B: Manual Download**
1. Go to **[TuSimple Dataset on Kaggle](https://www.kaggle.com/datasets/manideep1108/tusimple)**
2. Download and extract to `./data/kaggle/`

**Optional: Roboflow Dataset (for comparison experiments)**
1. Visit **[Road Mark Dataset on Roboflow](https://universe.roboflow.com/kip-8pf2a/road-mark-trjt6)**
2. Download in COCO format
3. Extract to `./data/roboflow/`

**Expected structure:**
```
data/kaggle/train_set/
├── clips/
│   ├── 0313-1/
│   ├── 0313-2/
│   ├── 0531/
│   └── 0601/
├── label_data_0313.json
├── label_data_0531.json
└── label_data_0601.json
```

### 3. Process Data (Convert Lane Points → Steering Angles)

```bash
python process_tusimple.py --path ./data/kaggle
```

**This will:**
- ✅ Read lane point annotations from JSON files
- ✅ Calculate steering angles using geometric algorithm
- ✅ Create train/val split (80/20)
- ✅ Generate CSVs: `tusimple_train_steering.csv`, `tusimple_val_steering.csv`
- ✅ Copy images to `data/processed/tusimple_images/`
- ✅ Display dataset statistics

**Expected output:**
```
Processing: 3,626 annotations
✓ Training samples: 2,901
✓ Validation samples: 725
✓ Mean steering: 0.0014 rad (0.08°)
✓ Distribution: 79.7% straight, 18.2% turns, 2.1% sharp turns
```

### 4. Run Exploratory Data Analysis

```python
from eda_analysis import DrivingDataEDA

eda = DrivingDataEDA(
    './data/processed/tusimple_images',
    './data/processed/tusimple_train_steering.csv',
    'tusimple'
)
eda.generate_full_report('./eda_results')
```

### 5. Train Model

```bash
# Basic training
python train.py --data_source tusimple --epochs 30 --batch_size 32

# With custom parameters
python train.py \
    --data_source tusimple \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --dropout 0.5 \
    --early_stopping 10
```

### 6. Monitor Training

```bash
tensorboard --logdir=./logs
# Open http://localhost:6006
```

### 7. Evaluate Best Model

```bash
python evaluate_best_model.py
```

---

## 💡 Lane-to-Steering Conversion Algorithm

### The Challenge
TuSimple provides lane waypoints (x-coordinates at specific y-heights) but no direct steering measurements.

### Our Solution

```python
def calculate_steering_from_lanes(lanes, h_samples):
    # 1. Focus on bottom 30% of image (immediate road ahead)
    target_y = int(image_height * 0.7)
    
    # 2. Get lane x-positions at target height
    lane_x_positions = [lane[target_y] for lane in lanes if valid]
    
    # 3. Calculate lane center
    lane_center = mean(lane_x_positions)
    
    # 4. Calculate offset from image center
    offset = lane_center - (image_width / 2)
    
    # 5. Convert to steering angle (±25° max)
    steering = -offset * 0.436 / (image_width / 2)
    
    return steering  # radians
```

**Key Features:**
- Uses bottom 30% of image for immediate path planning
- Handles multiple lanes (left, right, or both)
- Normalizes to ±0.436 radians (±25°)
- Robust to missing lane boundaries

---

## 📊 Model Architecture

### PilotNet (Modified NVIDIA Architecture)

```
Input: RGB Image (3 × 66 × 200)
    ↓
Conv1: 24 filters, 5×5, stride 2 → ELU → BatchNorm
Conv2: 36 filters, 5×5, stride 2 → ELU → BatchNorm
Conv3: 48 filters, 5×5, stride 2 → ELU → BatchNorm
Conv4: 64 filters, 3×3, stride 1 → ELU → BatchNorm
Conv5: 64 filters, 3×3, stride 1 → ELU → BatchNorm
    ↓
Flatten → 1152 features
    ↓
FC1: 1152 → 100 → ELU → Dropout(0.5)
FC2: 100 → 50 → ELU → Dropout(0.5)
FC3: 50 → 10 → ELU
Output: 10 → 1 (steering angle)
```

**Total Parameters:** ~252,000

---

## 📈 Results

### Baseline Performance (TuSimple Dataset)

| Metric | Value |
|--------|-------|
| **Training Samples** | 2,901 |
| **Validation Samples** | 725 |
| **Validation Loss (MSE)** | 0.0231 |
| **Validation MAE** | 0.0197 rad (1.13°) |
| **Accuracy (±3°)** | 72.4% |
| **Accuracy (±6°)** | 91.6% |
| **R² Score** | 0.8842 |

### Error Analysis by Steering Range

| Steering Range | Samples | MAE | Performance |
|----------------|---------|-----|-------------|
| Straight (-0.05 to 0.05) | 79.7% | 0.0156 rad | Best |
| Slight Turns (0.05 to 0.15) | 18.2% | 0.0287 rad | Good |
| Sharp Turns (>0.15) | 2.1% | 0.0534 rad | Needs improvement |

### Data Distribution

- **Straight driving:** 79.7% (highway bias)
- **Left turns:** 9.8%
- **Right turns:** 8.4%
- **Sharp turns:** 2.1%

---

## 🔧 Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_source` | `tusimple` | Dataset to use |
| `--epochs` | `50` | Training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--dropout` | `0.5` | Dropout rate |
| `--model` | `pilotnet` | Architecture |
| `--device` | `cuda` | Training device |
| `--early_stopping` | `10` | Early stop patience |
| `--num_workers` | `4` | Data loader workers (use 0 on Windows) |

---

## 📊 Evaluation Metrics

1. **Mean Absolute Error (MAE)**: Average steering prediction error
2. **Root Mean Squared Error (RMSE)**: Penalizes large errors
3. **R² Score**: Goodness of fit (1.0 = perfect)
4. **Accuracy Thresholds**: % predictions within ±3° and ±6°
5. **Error by Range**: Performance on straight vs turns

---

## 🎨 Data Augmentation

Applied during training (50% probability each):
- **Horizontal Flip**: Flips image and negates steering angle
- **Brightness Adjustment**: Random brightness variation
- **Random Shadow**: Adds realistic shadow effects

---

## 📅 Project Timeline

| Phase | Dates | Status |
|-------|-------|--------|
| Environment Setup & Data Collection | Sep 19 - Oct 3 | ✅ Complete |
| Model Development & Baseline Training | Oct 3 - Oct 17 | ✅ Complete |
| CARLA Data & Hybrid Training | Oct 17 - Nov 7 | 🔄 Planned |
| Results Analysis & Video 2 | Nov 7 - Nov 14 | 📅 Planned |
| Final Report & Presentation | Nov 14 - Dec 5 | 📅 Planned |

---

## 🎯 Milestone Achievements (Week 4)

✅ **Implementation Progress:**
- Complete data processing pipeline with novel conversion algorithm
- PilotNet architecture implemented and tested
- Training framework with TensorBoard logging
- Comprehensive evaluation metrics
- EDA and visualization tools

✅ **Results:**
- Baseline model: 1.13° MAE, 72.4% accuracy (±3°)
- Dataset: 3,626 samples (vs. original <1,000)
- Training curves showing good convergence
- Error analysis completed

✅ **Code Quality:**
- Clean, modular code structure
- Comprehensive documentation
- GitHub repository
- Reproducible experiments

---

## 🔬 Future Work

### Next Steps (Final Report):
1. **CARLA Synthetic Data**: Collect simulated driving data
2. **Domain Comparison**: Real (TuSimple) vs Synthetic (CARLA)
3. **Hybrid Training**: Combine real + synthetic data
4. **Sim-to-Real Transfer**: Analyze domain adaptation

### Potential Extensions:
- Obstacle detection and emergency braking
- Multi-task learning (lanes + objects)
- Temporal smoothing with LSTMs
- Real-time deployment on embedded systems

---

## 🐛 Troubleshooting

### Common Issues

**"CUDA Out of Memory"**
```bash
# Reduce batch size
python train.py --batch_size 16

# Or use CPU
python train.py --device cpu
```

**"No valid samples processed"**
- Check dataset structure matches expected format
- Verify JSON files exist: `label_data_*.json`
- Ensure `clips/` directory has images

**"Data loading errors on Windows"**
```bash
# Set num_workers to 0
python train.py --num_workers 0
```

**Training not converging**
- Check steering angle range (should be ~±0.4 rad)
- Visualize data to ensure correct loading
- Try lower learning rate: `--lr 1e-5`

---

## 💻 System Requirements

### Minimum:
- Python 3.8+
- 8GB RAM
- CPU (slower training)
- 25GB disk space

### Recommended:
- Python 3.9+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM (RTX 3060 Ti or better)
- CUDA 11.8+
- 50GB+ disk space

---

## 📚 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
opencv-python>=4.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tensorboard>=2.13.0
tqdm>=4.65.0
jupyter>=1.0.0
```

---

## 📖 References

[1] TuSimple. (2017). TuSimple Lane Detection Challenge.  
[2] Bojarski, M., et al. (2016). End to end learning for self-driving cars. *arXiv:1604.07316*  
[3] Dosovitskiy, A., et al. (2017). CARLA: An open urban driving simulator. *CoRL 2017*  
[4] Hu, C., et al. (2022). Sim-to-Real Domain Adaptation for Lane Detection. *arXiv:2202.07133*  
[5] Prakash, A., et al. (2019). Structured domain randomization. *ICRA 2019*

---

## ✨ Key Highlights

- 🎯 **Approach**: First implementation converting TuSimple lane points to steering angles
- 📊 **Strong Performance**: 72.4% accuracy within ±3° on challenging benchmark
- 🔬 **Reproducible**: Complete pipeline from raw data to trained model
- 📚 **Well-Documented**: Clean code with comprehensive documentation
- 🚀 **Scalable**: Easily adaptable to other lane detection datasets

**Project Status:** Milestone achieved ✅ | On track for final deliverables 📈