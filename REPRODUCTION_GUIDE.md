# Reproduction Guide

Step by step, from a fresh clone to the results table in the README.

> **Updated November 2026.** The original version of this guide listed "expected outputs" (real 72.4%, CARLA 59.1%, hybrid 90.0%) that came from a flawed evaluation and could not be reproduced. The expected results below come from `eval_clean.py` on TuSimple's official test set. Your exact numbers will differ slightly with seed, data and software versions; what should reproduce is the ranking.

**Time:** about 2–3 hours with a GPU, plus another 1–2 hours if you collect CARLA data.

---

## Prerequisites

- Python 3.10–3.12
- NVIDIA GPU with 8GB+ VRAM recommended. RTX 50-series cards need a PyTorch build for CUDA 12.8 or newer.
- About 50GB free disk space

---

## Step 1: Environment (5 minutes)

```bash
git clone https://github.com/kiprono1005/Lane-Detection-Capstone.git
cd Lane-Detection-Capstone

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
python test_setup.py
```

Check that PyTorch can see your GPU before training:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

If it prints `+cpu`, `None`, or `False`, reinstall PyTorch from the CUDA index, for example:

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## Step 2: Download TuSimple (10–30 minutes)

```bash
kaggle datasets download -d manideep1108/tusimple
unzip tusimple.zip -d ./data/kaggle
```

Or download manually from [Kaggle](https://www.kaggle.com/datasets/manideep1108/tusimple) and extract to `./data/kaggle/`.

You need both parts:

- `train_set/` with `clips/` and `label_data_0313.json`, `label_data_0531.json`, `label_data_0601.json` (training)
- `test_set/clips/` and `test_label.json` (evaluation only)

If the archive extracts into a `TUSimple` subfolder, use that folder wherever this guide says `./data/kaggle`.

---

## Step 3: Process TuSimple (5 minutes)

```bash
python process_tusimple.py --path ./data/kaggle
```

This converts lane points to steering labels and writes `./data/processed/`: 3,626 labeled frames, shuffled with seed 42 and split 80/20 into about 2,900 training and 725 validation frames.

---

## Step 4: Train the Real-Only Model (~20 minutes per seed)

```bash
python train.py --data_source tusimple --num_workers 0 --seed 42 --experiment_name pilotnet_tusimple_s42
python train.py --data_source tusimple --num_workers 0 --seed 1  --experiment_name pilotnet_tusimple_s1
```

One seed is enough to reproduce the main result; two show how much a retrain moves it.

---

## Step 5: Collect CARLA Data (optional, 1–2 hours)

Requires **CARLA 0.9.16** (Unreal Engine 4). CARLA 0.10.x does not include Town01–05, which the collector uses.

```bash
pip install carla==0.9.16    # must match the server version exactly

# Start the server and wait for the town to load
CarlaUE4.exe     # Windows
./CarlaUE4.sh    # Linux

# Separate terminal
python carla_data_collection.py --samples 3000 --output ./data/carla
```

Always pass `--output`; the script's default path points to the author's machine. Spot-check a few images in `./data/carla/images`: the sky should look blue. If it looks orange or brown, you have a version of the collector from before the color fix.

---

## Step 6: Train the CARLA and Mixed Models (~20–70 minutes each)

```bash
# CARLA only
python train.py --data_source carla --num_workers 0 --experiment_name pilotnet_carla_v2

# 50/50: 1,500 real + 1,500 CARLA, same total size as real-only
python create_hybrid_dataset.py --mode equal --total_samples 3000
python train.py --data_source hybrid --num_workers 0 --experiment_name pilotnet_hybrid_v2

# All real + all CARLA. Writes to ./data/hybrid, so keep the 50/50 copy first:
#   Windows:   Rename-Item data\hybrid hybrid_equal
#   Linux/Mac: mv data/hybrid data/hybrid_equal
python create_hybrid_dataset.py --mode augment
python train.py --data_source hybrid --num_workers 0 --experiment_name pilotnet_hybrid_augment
```

The "Best validation loss" printed at the end of the CARLA run will look very low. It's measured on CARLA's own random split, where frames a tenth of a second apart land on both sides, so don't treat it as a result.

---

## Step 7: Evaluate (a few minutes per run)

```bash
python eval_clean.py \
  --ckpt_real   checkpoints/pilotnet_tusimple_s42/best_model.pth \
  --ckpt_carla  checkpoints/pilotnet_carla_v2/best_model.pth \
  --ckpt_hybrid checkpoints/pilotnet_hybrid_v2/best_model.pth \
  --out results/clean_eval_v2

python eval_clean.py \
  --ckpt_real   checkpoints/pilotnet_tusimple_s42/best_model.pth \
  --ckpt_carla  checkpoints/pilotnet_carla_v2/best_model.pth \
  --ckpt_hybrid checkpoints/pilotnet_hybrid_augment/best_model.pth \
  --out results/clean_eval_augment
```

The output's opening line should read `TuSimple official test set: 2782 labeled frames`. If it says it fell back to the validation split, the script couldn't find `test_label.json` and `test_set/clips`; pass `--tusimple_root` pointing at the folder that contains them.

Each run saves `summary.csv`, `predictions.csv` and `run_info.json` to its `--out` folder. The hybrid row is labeled "Hybrid 50/50" whichever hybrid checkpoint you pass.

---

## Expected Results

TuSimple official test set, 2,782 frames. These are the author's November 2026 numbers.

| Model | MAE | Within ±3° |
|---|---|---|
| Real only (seed 1 / seed 42) | 1.15° / 1.23° | 93.7% / 92.9% |
| Real + all CARLA v2 | 1.30° | 91.8% |
| 50/50 real + CARLA v2 | 1.66° | 84.5% |
| CARLA v2 only | 2.22° | 70.7% |
| Baseline: always predict the mean | 1.94° | 78.3% |
| Baseline: always predict straight | 2.23° | 70.7% |

**What should reproduce:**

1. Real-only is best, clearly below both baselines.
2. The 50/50 mix is roughly 0.4–0.5° worse than real-only, with a 95% interval that excludes zero.
3. CARLA-only lands at about the "always straight" baseline.
4. Adding CARLA on top of all the real data gives no meaningful gain. The gap is small (0.07–0.15° in the author's runs), about the same size as the 0.08° difference between two real-only seeds.

If your seed-42 real-only run lands well away from 1.23°, check your PyTorch version and that `process_tusimple.py` processed all 3,626 frames.

---

## Monitoring Training (Optional)

```bash
tensorboard --logdir=./logs
```

---

## Troubleshooting

**"CUDA requested but not available, falling back to CPU."** Stop the run (CPU training takes hours) and fix the PyTorch install as in Step 1.

**`UnpicklingError: Weights only load failed` at the end of training.** PyTorch 2.6+ refuses to load checkpoints containing NumPy arrays by default. `training.py` loads them with `weights_only=False`; make sure you have the current version. The trained model is saved before this step, so nothing is lost.

**Windows data loading errors.** Use `--num_workers 0`.

**CARLA client can't connect.** The server must be running, and the `carla` Python package must match its version exactly.

---

## Time Estimate (GPU)

| Step | Time |
|---|---|
| Environment | 5 min |
| Download TuSimple | 10–30 min |
| Process TuSimple | 5 min |
| Real-only, per seed | ~20 min |
| CARLA collection | 1–2 hours |
| CARLA-only model | ~35 min |
| 50/50 model | ~20–35 min |
| Real + all CARLA model | 45–70 min |
| Each evaluation run | a few min |
