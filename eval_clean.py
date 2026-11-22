"""
eval_clean.py: re-score the three capstone models on frames none of them trained on.

Put this file in the root of Lane-Detection-Capstone (next to model.py) and run:

    python eval_clean.py

What it does differently from compare_all_models.py
  1. Augmentation is OFF. Every model sees exactly the same images and labels.
  2. The test frames are held out from every model:
       --split official (default if found): TuSimple's own test set, never used
         for training or checkpoint selection. Labels are made with the same
         lane-to-steering function used for training.
       --split val (fallback): tusimple_val_steering.csv, minus every real frame
         the hybrid model trained or validated on. Caveat: the real-only model's
         best checkpoint was chosen on this set, which flatters it slightly.
  3. Every metric comes from the same prediction arrays, and two trivial
     baselines are printed for scale: always predict 0, and always predict the
     training-set mean.
  4. A paired bootstrap gives a 95% interval on the MAE gap between models.

Outputs go to results/clean_eval/: summary.csv, predictions.csv, run_info.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import PilotNet                      # noqa: E402
from data_pipeline import DrivingDataset        # noqa: E402

RAD2DEG = 180.0 / np.pi
TOL3 = 0.05   # the repo's "±3°" threshold, in radians (2.86°)
TOL6 = 0.10   # the repo's "±6°" threshold, in radians (5.73°)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def find_checkpoint(ckpt_root: Path, tag: str):
    """Newest checkpoints/pilotnet_<tag>_*/best_model.pth, or None."""
    hits = sorted(ckpt_root.glob(f"pilotnet_{tag}_*/best_model.pth"))
    return hits[-1] if hits else None


def load_model(path: Path, device):
    model = PilotNet()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    return model, epoch


def same_transform_as_training():
    # Identical to create_dataloaders() in data_pipeline.py
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((66, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


@torch.no_grad()
def predict(model, loader, device):
    preds = []
    for images, _ in loader:
        preds.append(model(images.to(device)).squeeze(1).cpu().numpy())
    return np.concatenate(preds)


def metrics(pred, y):
    err = pred - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    var = float(np.var(y))
    r2 = float(1.0 - np.mean(err ** 2) / var) if var > 0 else float("nan")
    return {
        "mae_rad": mae, "mae_deg": mae * RAD2DEG,
        "rmse_rad": rmse, "rmse_deg": rmse * RAD2DEG,
        "r2": r2,
        "within_3deg_pct": float(np.mean(np.abs(err) < TOL3) * 100),
        "within_6deg_pct": float(np.mean(np.abs(err) < TOL6) * 100),
        "bias_deg": float(np.mean(err) * RAD2DEG),
    }


def bucket_mae(pred, y):
    """MAE in degrees per steering bucket. The three buckets cover every frame."""
    out = {}
    for name, mask in [
        ("left (< -0.05)", y < -0.05),
        ("straight (|y| <= 0.05)", np.abs(y) <= 0.05),
        ("right (> 0.05)", y > 0.05),
    ]:
        n = int(mask.sum())
        out[name] = (n, float(np.mean(np.abs(pred[mask] - y[mask])) * RAD2DEG) if n else float("nan"))
    return out


def paired_bootstrap_mae_gap(pred_a, pred_b, y, n_boot=5000, seed=0):
    """95% interval on MAE(a) - MAE(b) in degrees, resampling frames with replacement."""
    rng = np.random.default_rng(seed)
    ea, eb = np.abs(pred_a - y), np.abs(pred_b - y)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    gaps = (ea[idx].mean(axis=1) - eb[idx].mean(axis=1)) * RAD2DEG
    return float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


# ---------------------------------------------------------------------------
# building the test set
# ---------------------------------------------------------------------------
def _label_fn():
    """The repo's own lane-to-steering function, or an exact copy if it can't be imported."""
    try:
        from process_tusimple import TuSimpleProcessor
        proc = TuSimpleProcessor.__new__(TuSimpleProcessor)
        return lambda lanes, h: TuSimpleProcessor.calculate_steering_from_lanes(proc, lanes, h)
    except Exception:
        def fn(lanes, h_samples, image_width=1280, image_height=720):
            if not lanes or not h_samples:
                return None
            target_y = int(image_height * 0.7)
            h_idx = min(range(len(h_samples)), key=lambda i: abs(h_samples[i] - target_y))
            xs = [lane[h_idx] for lane in lanes if h_idx < len(lane) and lane[h_idx] != -2]
            if not xs:
                return None
            offset = (np.mean(xs) - image_width / 2.0) / (image_width / 2.0)
            return float(np.clip(-offset * 0.436, -0.436, 0.436))
        return fn


def build_official_test(tusimple_root: Path, out_csv: Path):
    """Label TuSimple's official test set with the repo's own lane-to-steering function."""
    label = _label_fn()

    labels = None
    for cand in [tusimple_root / "test_label.json", tusimple_root / "test_set" / "test_label.json"]:
        if cand.exists():
            labels = cand
            break
    if labels is None:
        found = list(tusimple_root.rglob("test_label.json"))
        labels = found[0] if found else None
    if labels is None:
        return None, None

    # raw_file looks like "clips/0530/1492626047222176976_0/20.jpg"
    image_root = None
    for cand in [labels.parent, labels.parent / "test_set", tusimple_root / "test_set"]:
        if (cand / "clips").exists():
            image_root = cand
            break
    if image_root is None:
        return None, None

    rows = []
    with open(labels) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            s = label(rec["lanes"], rec["h_samples"])
            if s is not None and (image_root / rec["raw_file"]).exists():
                rows.append({"image_path": rec["raw_file"], "steering_angle": s})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return image_root, len(rows)


def build_val_test(processed: Path, hybrid_dir: Path, out_csv: Path):
    val = pd.read_csv(processed / "tusimple_val_steering.csv")
    seen = set()
    for name in ["hybrid_train_steering.csv", "hybrid_val_steering.csv"]:
        p = hybrid_dir / name
        if p.exists():
            paths = pd.read_csv(p)["image_path"].dropna().astype(str)
            seen |= {s[len("real_"):] for s in paths if s.startswith("real_")}
    clean = val[~val["image_path"].astype(str).isin(seen)]
    clean[["image_path", "steering_angle"]].to_csv(out_csv, index=False)
    return processed / "tusimple_images", len(val), len(clean)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["auto", "official", "val"], default="auto")
    ap.add_argument("--tusimple_root", default="./data/kaggle")
    ap.add_argument("--processed_dir", default="./data/processed")
    ap.add_argument("--hybrid_dir", default="./data/hybrid")
    ap.add_argument("--checkpoints", default="./checkpoints")
    ap.add_argument("--ckpt_real"); ap.add_argument("--ckpt_carla"); ap.add_argument("--ckpt_hybrid")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out", default="./results/clean_eval")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_csv = out / "test_set.csv"

    # 1. test set
    split, note = None, ""
    if args.split in ("auto", "official"):
        image_root, n = build_official_test(Path(args.tusimple_root), test_csv)
        if image_root is not None and n:
            split, note = "official", f"TuSimple official test set: {n} labeled frames, unseen by all three models."
        elif args.split == "official":
            sys.exit("Could not find test_label.json and test_set/clips under --tusimple_root.")
    if split is None:
        image_root, n_all, n = build_val_test(Path(args.processed_dir), Path(args.hybrid_dir), test_csv)
        split = "val"
        note = (f"tusimple_val minus hybrid frames: {n} of {n_all} frames kept. "
                "The real-only checkpoint was selected on this set, so its score is slightly flattered.")
    print(f"\nTEST SET: {note}")

    dataset = DrivingDataset(str(image_root), str(test_csv), transform=same_transform_as_training(),
                             augment=False, data_source="real")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    y = pd.read_csv(test_csv)["steering_angle"].to_numpy(dtype=np.float32)

    # 2. models
    ck_root = Path(args.checkpoints)
    paths = {
        "Real (TuSimple)": Path(args.ckpt_real) if args.ckpt_real else find_checkpoint(ck_root, "tusimple"),
        "CARLA (synthetic)": Path(args.ckpt_carla) if args.ckpt_carla else find_checkpoint(ck_root, "carla"),
        "Hybrid 50/50": Path(args.ckpt_hybrid) if args.ckpt_hybrid else find_checkpoint(ck_root, "hybrid"),
    }
    preds, info = {}, {}
    for name, p in paths.items():
        if p is None or not p.exists():
            print(f"  ! no checkpoint for {name}, skipping"); continue
        model, epoch = load_model(p, device)
        preds[name] = predict(model, loader, device)
        info[name] = {"checkpoint": str(p), "epoch": epoch}
        print(f"  scored {name}  ({p})")

    # 3. baselines, for scale
    train_mean = float(pd.read_csv(Path(args.processed_dir) / "tusimple_train_steering.csv")["steering_angle"].mean())
    preds["Baseline: always 0"] = np.zeros_like(y)
    preds["Baseline: train mean"] = np.full_like(y, train_mean)

    # 4. report
    rows = []
    for name, pr in preds.items():
        rows.append({"model": name, "n": len(y), **metrics(pr, y)})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print("\n" + "=" * 100)
    print(f"CLEAN EVALUATION  ({split} split, augmentation off, label std = {np.std(y) * RAD2DEG:.2f} deg)")
    print("=" * 100)
    show = table[["model", "mae_deg", "rmse_deg", "r2", "within_3deg_pct", "within_6deg_pct", "bias_deg"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    print("\nMAE by steering bucket (degrees, frame count in brackets)")
    for name, pr in preds.items():
        if name.startswith("Baseline"):
            continue
        parts = "   ".join(f"{k}: {v:.2f} [{n}]" for k, (n, v) in bucket_mae(pr, y).items())
        print(f"  {name:20s} {parts}")

    if "Hybrid 50/50" in preds and "Real (TuSimple)" in preds:
        lo, hi = paired_bootstrap_mae_gap(preds["Real (TuSimple)"], preds["Hybrid 50/50"], y)
        verdict = "hybrid is better" if lo > 0 else ("real is better" if hi < 0 else "not distinguishable")
        print(f"\nMAE(real) - MAE(hybrid): 95% interval [{lo:.3f}, {hi:.3f}] deg  ->  {verdict}")

    table.to_csv(out / "summary.csv", index=False)
    pd.DataFrame({"y": y, **{k: v for k, v in preds.items()}}).to_csv(out / "predictions.csv", index=False)
    with open(out / "run_info.json", "w") as f:
        json.dump({"split": split, "note": note, "device": device, "models": info,
                   "train_mean_rad": train_mean}, f, indent=2)
    print(f"\nSaved summary.csv, predictions.csv, run_info.json to {out}")


if __name__ == "__main__":
    main()
