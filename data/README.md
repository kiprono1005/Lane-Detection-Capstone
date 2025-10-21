# Dataset Setup Instructions

**Project:** Autonomous Lane Keeping Using Machine Learning  
**GitHub:** https://github.com/kiprono1005/Lane-Detection-Capstone

---

## 📁 Required Data Structure

```
data/
├── kaggle/
│   └── train_set/              # TuSimple dataset
│       ├── clips/
│       ├── label_data_0313.json
│       ├── label_data_0531.json
│       └── label_data_0601.json
├── processed/
│   ├── tusimple_images/
│   ├── tusimple_train_steering.csv
│   └── tusimple_val_steering.csv
└── README.md
```

## 📊 Primary Dataset: TuSimple

**Source:** https://www.kaggle.com/datasets/manideep1108/tusimple  
**Size:** ~21 GB  
**Samples:** 3,626 training images

### Download:
```bash
kaggle datasets download -d manideep1108/tusimple
unzip tusimple.zip -d ./data/kaggle
```

### Process:
```bash
python process_tusimple.py --path ./data/kaggle
```

## 🎨 Optional: Roboflow Dataset

**Source:** https://universe.roboflow.com/kip-8pf2a/road-mark-trjt6

---

For full instructions, see main README.