# Autonomous Lane Keeping: Real vs Synthetic vs Hybrid Training Data

**A controlled comparison of real, simulated, and mixed training data for end-to-end steering prediction**

**Author:** Kip Chemweno
**Course:** ECE 4424 - Machine Learning Capstone
**Institution:** Virginia Tech
**Date:** December 2025 (results corrected November 2026)
**GitHub:** https://github.com/kiprono1005/Lane-Detection-Capstone

---

> ### ⚠️ Results correction (November 2026)
>
> The results originally published here in December 2025 (hybrid 90.0% within ±3°, 1.43° MAE, "24.3% improvement over real-only") **did not hold up** under a clean re-evaluation. The original comparison applied data augmentation at test time, scored models on frames some of them had trained on, and reported no baselines.
>
> All numbers below come from a corrected evaluation (`eval_clean.py`) on **TuSimple's official test set: 2,782 frames that no model saw during training or checkpoint selection**, with augmentation off. **The headline finding reversed: real data alone performed best, and simulated data never improved accuracy.**

---

## 📋 Summary

This project asks whether simulated driving data can stand in for, or add to, real data when training a network to predict steering from a single camera image. The same PilotNet CNN was trained on real highway frames (TuSimple), simulated frames (CARLA), and two kinds of mixes, with all other settings held constant, then scored on real held-out frames.

### 🏆 Key Results

TuSimple official test set, 2,782 frames. "Within ±3°" means |error| < 0.05 rad (2.86°).

| Training data | Real frames in training | MAE | Within ±3° | R² |
|---|---|---|---|---|
| **Real only (seed 1)** | ~2,900 | **1.15°** | **93.7%** | **0.46** |
| **Real only (seed 42)** | ~2,900 | **1.23°** | **92.9%** | **0.42** |
| Real + all CARLA v2 (added on top) | ~2,900 | 1.30° | 91.8% | 0.39 |
| 50/50 real + CARLA v2 | ~1,200 | 1.66° | 84.5% | 0.19 |
| 50/50 real + CARLA v1 | ~1,200 | 1.75° | 83.2% | 0.13 |
| CARLA only, v2 | 0 | 2.22° | 70.7% | −0.20 |
| CARLA only, v1 | 0 | 3.29° | 54.9% | −1.64 |
| *Baseline: always predict the training mean* | — | *1.94°* | *78.3%* | *−0.02* |
| *Baseline: always predict straight (0°)* | — | *2.23°* | *70.7%* | *−0.18* |

CARLA **v1** is the original synthetic dataset; **v2** was re-collected after fixing a color-channel bug in the collector (see below).

**Main finding: accuracy tracked how much real data the model saw. Simulated data never helped.**

- Real-only reached **1.15–1.23° MAE across two training seeds**, about 40% less error than always predicting the mean angle. Retraining with seed 42 reproduced the original December 2025 checkpoint almost exactly.
- **Swapping half the real data for simulated data** (50/50 mix) cost about 0.4–0.5°. Paired bootstrap, 95% interval on MAE(real) − MAE(mix): −0.56° to −0.48° (v1), −0.47° to −0.39° (v2).
- **Adding simulated data on top of all the real data** gave no benefit: 1.30° vs 1.15–1.23°. Real-only was ahead under both seeds (by 0.07° and 0.15°), but the two real-only seeds differ from each other by 0.08° and the combined model was trained once, so this is reported as "no benefit," not "harmful."
- **Simulated data alone** was no better than always predicting straight, even after the color fix.

### Error by steering direction (MAE in degrees)

| Model | Left (< −0.05 rad), n=670 | Straight, n=1,966 | Right (> 0.05 rad), n=146 |
|---|---|---|---|
| Real only (seed 42) | 1.70 | 0.80 | 4.85 |
| Real + all CARLA v2 | 2.04 | 0.78 | 4.84 |
| 50/50 real + CARLA v2 | 2.61 | 1.03 | 5.76 |
| CARLA only, v2 | 4.04 | 1.31 | 6.14 |

Right turns are the weak spot for every model; they are also the rarest case in the data. Adding CARLA on top of real data changed nothing on straights or right turns; its whole gap is on left turns, and its predictions drifted about 0.3° toward straight on average.

---

## 🔍 Why simulated data didn't help

1. **The two sources label different quantities.** The TuSimple label is a *lateral offset*: where the lane-marking center sits relative to the image center. The CARLA label is a *steering command* aimed at a waypoint 5 m ahead. They share a ±25° scale but not a meaning.
2. **The simulated car never left the lane center.** It followed center waypoints, so the synthetic data never showed an off-center vehicle, which is mostly what the real label measures. A model trained on it learns to predict "straight." The original PilotNet work (Bojarski et al., 2016) addressed the same issue with offset side cameras and shifted labels to teach recovery.
3. **Color-channel bug (fixed in v2).** CARLA delivers camera frames as BGRA. The v1 collector kept the BGR channels and then converted "RGB to BGR" before saving, swapping red and blue. Fixing it and re-collecting cut CARLA-only error by a third (3.29° → 2.22°), but only to the level of the do-nothing baseline. Note that v2 also used fresh driving routes, so not all of that change is attributable to the fix.
4. **Visual and scenario gap.** Simulated textures and lighting versus real cameras; CARLA's urban towns versus TuSimple's highways.
5. **Image–label timing.** The collector runs CARLA asynchronously, so the saved frame and the recorded steering command can be up to a frame apart.

## 🧾 What was wrong with the original evaluation

- **Augmentation at test time.** `compare_all_models.py` built its test loader from an augmented dataset, so test images were randomly flipped, brightened and shaded, and each model saw different inputs.
- **Test frames overlapped training data.** The test set was a 20% slice of `tusimple_train_steering.csv`, which the real-only model trained on in full; the hybrid set drew real frames from train and validation combined.
- **No baselines.** About 76% of TuSimple labels are within ±3° of zero, so threshold accuracy means little without a "predict straight" reference.
- **Checkpoint selection.** The comparison script took the first matching checkpoint, unsorted, which may not be the newest run.
- **The published table was internally inconsistent** (its RMSE and R² values implied a label spread several times wider than the data's).

`eval_clean.py` fixes all of these. The original `compare_all_models.py` is kept for reference.

## 🔭 Next steps

- Compute CARLA labels the same way as TuSimple labels (lateral offset from lane geometry).
- Drive off-center and record recovery behavior, or add offset virtual cameras.
- Collect in CARLA synchronous mode so each label matches its frame.
- Train every configuration with several seeds and report the spread.
- Split CARLA data by town or drive rather than by frame, so near-duplicate neighboring frames don't land in both training and validation.
- Fix the TuSimple label to use the two markings nearest the image center, rather than averaging every visible marking.

---

## 🚀 Quick Start Guide

### Prerequisites

- Python 3.10–3.12
- CUDA-capable GPU (recommended, 8GB+ VRAM). RTX 50-series cards need a PyTorch build for CUDA 12.8 or newer.
- 50GB free disk space

### 1. Clone and Set Up the Environment

```bash
git clone https://github.com/kiprono1005/Lane-Detection-Capstone.git
cd Lane-Detection-Capstone

python -m venv venv
source venv/bin/activate   # Linux/Mac
# OR
venv\Scripts\activate      # Windows

pip install -r requirements.txt
python test_setup.py
```

### 2. Download Datasets

#### TuSimple (Real-World Data) - Required

```bash
kaggle datasets download -d manideep1108/tusimple
unzip tusimple.zip -d ./data/kaggle
```

Or download manually from [TuSimple on Kaggle](https://www.kaggle.com/datasets/manideep1108/tusimple) and extract to `./data/kaggle/`. The download includes the official test set (`test_label.json` and `test_set/clips/`) used for evaluation.

#### CARLA (Synthetic Data) - Optional

Requires CARLA **0.9.16** (Unreal Engine 4). CARLA 0.10.x does not include Town01–05.

```bash
pip install carla==0.9.16

# Start the server
CarlaUE4.exe     # Windows
./CarlaUE4.sh    # Linux

# Collect data (separate terminal)
python carla_data_collection.py --samples 3000 --output ./data/carla
```

### 3. Data Processing

```bash
python process_tusimple.py --path ./data/kaggle
# 3,626 labeled frames, split 80/20 by frame (seed 42): ~2,900 train / ~725 validation
```

If the archive extracted into a `TUSimple` subfolder, point `--path` at the folder that contains `train_set`.

### 4. Training

```bash
# Real only
python train.py --data_source tusimple --num_workers 0 --experiment_name pilotnet_tusimple_s42

# CARLA only
python train.py --data_source carla --num_workers 0 --experiment_name pilotnet_carla_v2

# 50/50 mix (1,500 real + 1,500 CARLA)
python create_hybrid_dataset.py --mode equal --total_samples 3000
python train.py --data_source hybrid --num_workers 0 --experiment_name pilotnet_hybrid_v2

# All real + all CARLA (overwrites data/hybrid; rename the 50/50 folder first to keep it)
python create_hybrid_dataset.py --mode augment
python train.py --data_source hybrid --num_workers 0 --experiment_name pilotnet_hybrid_augment
```

Use `--seed` to train additional seeds. About 15–35 minutes per model on an RTX 5070.

### 5. Evaluation

```bash
python eval_clean.py \
  --ckpt_real   checkpoints/pilotnet_tusimple_s42/best_model.pth \
  --ckpt_carla  checkpoints/pilotnet_carla_v2/best_model.pth \
  --ckpt_hybrid checkpoints/pilotnet_hybrid_v2/best_model.pth \
  --out results/clean_eval_v2
```

`eval_clean.py` labels TuSimple's official test set with the same lane-to-steering function used in training, scores every model with augmentation off, prints MAE, RMSE, R² and threshold accuracy next to two do-nothing baselines, breaks error down by steering direction, and gives a paired-bootstrap 95% interval on the real-versus-hybrid MAE gap. If the official test set is missing, it falls back to the validation split minus every frame the hybrid model used.

### 6. Monitor Training (Optional)

```bash
tensorboard --logdir=./logs
```

---

## 🔬 Methodology

### Lane-to-Steering Conversion

TuSimple provides lane-marking points, not steering. `process_tusimple.py` converts them:

```python
def calculate_steering_from_lanes(lanes, h_samples):
    target_y = int(image_height * 0.7)                 # y ≈ 504 of 720
    xs = [lane[target_y] for lane in lanes if valid(lane, target_y)]
    lane_center = mean(xs)                             # all visible markings
    offset = (lane_center - image_width / 2) / (image_width / 2)
    return clip(-offset * 0.436, -0.436, 0.436)        # ±25°
```

This is a lateral-offset label rather than true steering: it ignores heading, curvature and speed. It also averages every visible lane marking, so an unequal number of markings on each side shifts the label even when the car is centered.

### Model Architecture: Modified PilotNet

```
Input: RGB (3×66×200), normalized to [-1, 1]
Conv1: 24@5×5, stride 2 → BatchNorm → ELU     (24×31×98)
Conv2: 36@5×5, stride 2 → BatchNorm → ELU     (36×14×47)
Conv3: 48@5×5, stride 2 → BatchNorm → ELU     (48×5×22)
Conv4: 64@3×3, stride 1 → BatchNorm → ELU     (64×3×20)
Conv5: 64@3×3, stride 1 → BatchNorm → ELU     (64×1×18)
Flatten → 1,152
FC1: 1,152 → 100 → ELU → Dropout(0.5)
FC2: 100 → 50 → ELU → Dropout(0.5)
FC3: 50 → 10 → ELU
Output: 10 → 1 (steering angle, radians)

Total parameters: 252,691
```

### Datasets

| Dataset | Size | Notes |
|---|---|---|
| TuSimple (real) | 3,626 labeled frames, 1280×720 | US highways; one labeled frame per 1-second clip; ~2,900 train / ~725 val |
| TuSimple official test set | 2,782 labeled frames | Used only for evaluation |
| CARLA (synthetic) | ~3,000 frames | Town01–05; clear noon, cloudy noon, wet noon, clear sunset; 1280×720 camera, 90° FOV, 1.5 m forward, 1.4 m up, pitched 5° down; waypoint-following driver at ~20 km/h |
| Hybrid, equal mode | 1,500 real + 1,500 CARLA | Same total size as real-only; stratified 80/20 split |
| Hybrid, augment mode | all real + all CARLA (~6,600) | Same real data as real-only, plus CARLA |

### Training Protocol

- Optimizer: Adam (LR 1e-4, weight decay 1e-5); gradient clipping at norm 1.0
- Scheduler: ReduceLROnPlateau (factor 0.5, patience 5)
- Loss: mean squared error
- Batch size 32, up to 30 epochs, early stopping after 10 epochs without improvement
- Best checkpoint chosen by validation loss
- Augmentation (training only, each with 50% probability): horizontal flip with negated angle, random brightness (25–125%), random shadow

**Held constant across models:** architecture, hyperparameters, augmentation, and the evaluation set (TuSimple official test set).

---

## 📂 Project Structure

```
Lane-Detection-Capstone/
├── data/                         # not committed
│   ├── kaggle/                   # TuSimple raw data, incl. test_set/ and test_label.json
│   ├── carla/                    # CARLA collected data
│   ├── hybrid/                   # mixed datasets
│   └── processed/                # TuSimple frames + steering CSVs
├── checkpoints/                  # trained models (not committed)
├── results/                      # evaluation outputs
├── logs/                         # TensorBoard logs
├── eda_results/                  # exploratory data analysis plots
│
├── model.py                      # PilotNet architecture
├── train.py                      # training entry point
├── training.py                   # training loop
├── data_pipeline.py              # data loading and augmentation
├── evaluation.py                 # metrics
├── eval_clean.py                 # corrected held-out evaluation (use this)
├── compare_all_models.py         # original comparison (superseded)
├── evaluate_best_model.py        # single-model evaluation
│
├── process_tusimple.py           # TuSimple lanes → steering labels
├── carla_data_collection.py      # CARLA data collector
├── create_hybrid_dataset.py      # equal / real_heavy / augment mixes
├── lane_to_steering.py           # converters for other lane datasets
│
├── eda_analysis.py
├── visualization_utils.py
├── test_setup.py
└── requirements.txt
```

---

## 🛠️ Troubleshooting

**"CUDA requested but not available."** The environment has a CPU-only PyTorch build. Reinstall from the PyTorch index for your CUDA version, e.g. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`.

**`UnpicklingError: Weights only load failed`.** PyTorch 2.6+ defaults to `weights_only=True`. Checkpoints here store NumPy arrays, so they load with `weights_only=False` (already set in `training.py` and `eval_clean.py`). Only do this for checkpoints you trust.

**Data loading errors on Windows.** Use `--num_workers 0`.

**CARLA connection issues.** Make sure the server is running and the client version matches it exactly (`pip install carla==0.9.16`).

---

## 💻 My Setup

- GPU: NVIDIA RTX 5070 (12GB)
- CPU: AMD Ryzen 7 7600X
- RAM: 64GB DDR5
- OS: Windows 11

---

## 📚 Dependencies

```txt
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
Pillow>=10.0.0
```

---

## 📄 References

[1] M. Bojarski et al., "End to end learning for self-driving cars," *arXiv:1604.07316*, 2016.

[2] A. Dosovitskiy et al., "CARLA: An open urban driving simulator," *CoRL*, 2017.

[3] F. Codevilla et al., "End-to-end driving via conditional imitation learning," *ICRA*, 2018.

[4] C. Hu et al., "Sim-to-real domain adaptation for lane detection," *arXiv:2202.07133*, 2022.

[5] A. Prakash et al., "Structured domain randomization," *ICRA*, 2019.

[6] TuSimple, "TuSimple Lane Detection Challenge," 2017. [Dataset](https://www.kaggle.com/datasets/manideep1108/tusimple)

---

**Status:** Complete · Capstone project, December 2025 · Results corrected November 2026
