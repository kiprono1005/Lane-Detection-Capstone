"""
Test Script to Verify Installation and Setup
Run this to ensure everything is working before starting training

Usage:
    python test_setup.py

Author: Kip Chemweno
Course: ECE 4424 - Machine Learning
Date: October 2025
"""

import sys

def check_python_version():
    """Check Python version."""
    print("\n1. Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ✗ Python {version.major}.{version.minor}.{version.micro}")
        print("   Required: Python 3.8+")
        return False

def check_packages():
    """Check required packages."""
    print("\n2. Checking required packages...")

    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'cv2': 'OpenCV',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'Scikit-learn',
        'tqdm': 'tqdm'
    }

    all_installed = True

    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"   ✓ {name}")
        except ImportError:
            print(f"   ✗ {name} - NOT INSTALLED")
            all_installed = False

    return all_installed

def check_cuda():
    """Check CUDA availability."""
    print("\n3. Checking CUDA/GPU support...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"   ✓ CUDA available")
            print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"   ✓ CUDA version: {torch.version.cuda}")
            total_memory = torch.cuda.get_device_properties(0).total_memory
            print(f"   ✓ GPU Memory: {total_memory / 1e9:.2f} GB")
            return True
        else:
            print(f"   ⚠  CUDA not available - will use CPU")
            print(f"   Note: Training will be slower on CPU")
            return True  # Not a failure, just slower
    except Exception as e:
        print(f"   ✗ Error checking CUDA: {e}")
        return False

def test_model_creation():
    """Test model instantiation."""
    print("\n4. Testing model creation...")

    try:
        from model import PilotNet
        import torch

        model = PilotNet()
        print(f"   ✓ PilotNet model created")

        # Test forward pass
        dummy_input = torch.randn(1, 3, 66, 200)
        output = model(dummy_input)

        if output.shape == (1, 1):
            print(f"   ✓ Model forward pass successful")
            print(f"   ✓ Output shape: {output.shape}")
            return True
        else:
            print(f"   ✗ Unexpected output shape: {output.shape}")
            return False

    except Exception as e:
        print(f"   ✗ Error creating model: {e}")
        return False

def test_data_pipeline():
    """Test data pipeline modules."""
    print("\n5. Testing data pipeline...")

    try:
        from data_pipeline import DrivingDataset, ImagePreprocessor
        import numpy as np

        # Test image preprocessor
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        processed = ImagePreprocessor.preprocess_image(dummy_image)

        if processed.shape == (66, 200, 3):
            print(f"   ✓ Image preprocessing working")
        else:
            print(f"   ✗ Unexpected processed shape: {processed.shape}")
            return False

        # Test ROI crop
        cropped = ImagePreprocessor.crop_roi(dummy_image)
        print(f"   ✓ ROI cropping working")

        print(f"   ✓ Data pipeline modules loaded")
        return True

    except Exception as e:
        print(f"   ✗ Error testing data pipeline: {e}")
        return False

def test_training_module():
    """Test training utilities."""
    print("\n6. Testing training module...")

    try:
        from training import Trainer
        print(f"   ✓ Trainer class loaded")
        return True
    except Exception as e:
        print(f"   ✗ Error loading training module: {e}")
        return False

def test_evaluation_module():
    """Test evaluation utilities."""
    print("\n7. Testing evaluation module...")

    try:
        from evaluation import ModelEvaluator
        print(f"   ✓ ModelEvaluator class loaded")
        return True
    except Exception as e:
        print(f"   ✗ Error loading evaluation module: {e}")
        return False

def test_processing_modules():
    """Test lane-to-steering conversion modules."""
    print("\n8. Testing processing modules...")

    try:
        # TuSimple is your main processor
        from process_tusimple import TuSimpleProcessor
        print(f"   ✓ TuSimpleProcessor loaded (main)")

        # Optional processors
        try:
            from lane_to_steering import LaneToSteering, RoboflowDatasetProcessor
            print(f"   ✓ RoboflowDatasetProcessor loaded (optional)")
        except ImportError:
            print(f"   ℹ  Roboflow processor not available (optional)")

        return True
    except Exception as e:
        print(f"   ✗ Error loading processing modules: {e}")
        return False

def check_directory_structure():
    """Check if required directories exist."""
    print("\n9. Checking directory structure...")

    from pathlib import Path

    required_dirs = {
        'data': 'Data directory',
        'checkpoints': 'Model checkpoints directory',
        'logs': 'TensorBoard logs directory',
        'results': 'Results directory'
    }

    all_exist = True

    for dir_name, description in required_dirs.items():
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"   ✓ {description}: {dir_path}")
        else:
            print(f"   ⚠  {description} does not exist - will be created automatically")
            # Create the directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"     Created: {dir_path}")

    return True

def check_raw_data_availability():
    """Check if raw lane detection datasets are available."""
    print("\n10. Checking raw dataset availability...")

    from pathlib import Path

    datasets_found = []

    # Check for TuSimple dataset (from Kaggle)
    tusimple_dir = Path('data/kaggle/train_set')
    if tusimple_dir.exists():
        label_files = list(tusimple_dir.glob('label_data_*.json'))
        if label_files:
            # Count images by checking clips directory
            clips_dir = tusimple_dir / 'clips'
            if clips_dir.exists():
                num_images = len(list(clips_dir.rglob('*.jpg')))
                print(f"   ✓ TuSimple dataset found (Kaggle)")
                print(f"     - {len(label_files)} label JSON files")
                print(f"     - ~{num_images} training images")
                datasets_found.append('tusimple')
            else:
                print(f"   ✓ TuSimple dataset found (Kaggle)")
                print(f"     - {len(label_files)} label JSON files")
                datasets_found.append('tusimple')
        else:
            print(f"   ⚠  TuSimple label files not found in {tusimple_dir}")
    else:
        print(f"   ⚠  TuSimple dataset not found: {tusimple_dir}")
        print(f"      Expected structure: data/kaggle/train_set/label_data_*.json")

    # Check for Roboflow dataset (optional)
    roboflow_dir = Path('data/roboflow')
    if roboflow_dir.exists():
        train_dir = roboflow_dir / 'train'
        if train_dir.exists():
            annotations = train_dir / '_annotations.coco.json'
            if annotations.exists():
                num_images = len(list(train_dir.glob('*.jpg'))) + len(list(train_dir.glob('*.png')))
                print(f"   ✓ Roboflow dataset found (optional)")
                print(f"     - {num_images} training images")
                print(f"     - COCO annotations present")
                datasets_found.append('roboflow')
            else:
                print(f"   ⚠  Roboflow annotations not found")
        else:
            print(f"   ⚠  Roboflow train directory not found")
    else:
        print(f"   ℹ  Roboflow dataset not found (optional): {roboflow_dir}")

    if not datasets_found:
        print(f"\n   ⚠  No raw datasets found!")
        print(f"   You need to download the TuSimple dataset from Kaggle:")
        print(f"   - Download from: https://www.kaggle.com/datasets/manideep1108/tusimple")
        print(f"   - Extract to: data/kaggle/train_set/")
        print(f"   - Expected structure:")
        print(f"       data/kaggle/train_set/")
        print(f"       ├── clips/")
        print(f"       │   └── [video clip folders with .jpg images]")
        print(f"       ├── label_data_0313.json")
        print(f"       ├── label_data_0531.json")
        print(f"       └── label_data_0601.json")
        return False
    else:
        print(f"\n   ✓ Found {len(datasets_found)} dataset(s): {', '.join(datasets_found)}")
        return True

def check_processed_data_availability():
    """Check if processed data is available."""
    print("\n11. Checking processed data availability...")

    from pathlib import Path

    processed_dir = Path('data/processed')

    if not processed_dir.exists():
        print(f"   ⚠  Processed data directory not found: {processed_dir}")
        print(f"   Run lane-to-steering conversion first:")
        print(f"   - python process_tusimple.py --path ./data/kaggle")
        print(f"   - python lane_to_steering.py --dataset roboflow --path ./data/roboflow --splits train valid (if using Roboflow)")
        return False

    datasets_found = []

    # Check for processed TuSimple data (your main dataset)
    tusimple_csv = processed_dir / 'tusimple_train_steering.csv'
    tusimple_val_csv = processed_dir / 'tusimple_val_steering.csv'
    tusimple_images = processed_dir / 'tusimple_images'
    if tusimple_csv.exists() and tusimple_images.exists():
        import pandas as pd
        df_train = pd.read_csv(tusimple_csv)
        num_images = len(list(tusimple_images.glob('*.jpg')))
        print(f"   ✓ Processed TuSimple data found")
        print(f"     - {len(df_train)} training samples in CSV")

        if tusimple_val_csv.exists():
            df_val = pd.read_csv(tusimple_val_csv)
            print(f"     - {len(df_val)} validation samples in CSV")

        print(f"     - {num_images} images")
        datasets_found.append('tusimple')
    else:
        print(f"   ⚠  Processed TuSimple data not found")
        print(f"      Missing: {tusimple_csv} or {tusimple_images}")

    # Check for processed Roboflow data (optional)
    roboflow_csv = processed_dir / 'roboflow_train_steering.csv'
    roboflow_images = processed_dir / 'roboflow_images'
    if roboflow_csv.exists() and roboflow_images.exists():
        import pandas as pd
        df = pd.read_csv(roboflow_csv)
        num_images = len(list(roboflow_images.glob('*.jpg'))) + len(list(roboflow_images.glob('*.png')))
        print(f"   ✓ Processed Roboflow data found (optional)")
        print(f"     - {len(df)} samples in CSV")
        print(f"     - {num_images} images")
        datasets_found.append('roboflow')
    else:
        print(f"   ℹ  Processed Roboflow data not found (optional)")

    if not datasets_found:
        print(f"\n   ⚠  No processed data found!")
        print(f"   Process your TuSimple data first:")
        print(f"   - python process_tusimple.py --path ./data/kaggle")
        print(f"\n   This will:")
        print(f"   - Read lane point annotations from JSON files")
        print(f"   - Calculate steering angles from lane positions")
        print(f"   - Create training and validation CSVs")
        print(f"   - Copy images to data/processed/tusimple_images/")
        return False
    else:
        print(f"\n   ✓ Found {len(datasets_found)} processed dataset(s): {', '.join(datasets_found)}")
        return True

def print_summary(results):
    """Print summary of all checks."""
    print("\n" + "="*70)
    print("SETUP VERIFICATION SUMMARY")
    print("="*70)

    total_checks = len(results)
    passed_checks = sum(results.values())

    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} - {check_name}")

    print("="*70)
    print(f"Results: {passed_checks}/{total_checks} checks passed")
    print("="*70)

    if passed_checks == total_checks:
        print("\n🎉 All checks passed! You're ready to start training.")
        print("\nNext steps:")
        print("  1. If data not processed: python process_tusimple.py --path ./data/kaggle")
        print("  2. Run EDA: python -c \"from eda_analysis import DrivingDataEDA; eda = DrivingDataEDA('./data/processed/tusimple_images', './data/processed/tusimple_train_steering.csv', 'tusimple'); eda.generate_full_report()\"")
        print("  3. Train baseline: python train.py --data_source tusimple --epochs 30")
        print("  4. Monitor training: tensorboard --logdir=./logs")
    else:
        print("\n⚠️  Some checks failed. Please address the issues above before training.")

        if not results.get('Raw data availability', True):
            print("\n📁 DATA DOWNLOAD GUIDE:")
            print("   Download the TuSimple dataset from Kaggle:")
            print("   - URL: https://www.kaggle.com/datasets/manideep1108/tusimple")
            print("   - Extract to: data/kaggle/train_set/")
            print("   - Structure:")
            print("       data/kaggle/train_set/")
            print("       ├── clips/")
            print("       ├── label_data_0313.json")
            print("       ├── label_data_0531.json")
            print("       └── label_data_0601.json")

        if not results.get('Processed data availability', True):
            print("\n🔧 DATA PROCESSING GUIDE:")
            print("   Run the processing script to convert lane labels to steering angles:")
            print("   - TuSimple: python process_tusimple.py --path ./data/kaggle")
            print("\n   This will create:")
            print("   - data/processed/tusimple_train_steering.csv")
            print("   - data/processed/tusimple_val_steering.csv")
            print("   - data/processed/tusimple_images/ (with all training images)")
            print("\n   Then you can train with:")
            print("   - python train.py --data_source tusimple --epochs 30")

def main():
    """Run all setup checks."""
    print("="*70)
    print("AUTONOMOUS LANE KEEPING - SETUP VERIFICATION")
    print("="*70)
    print("This script will verify your environment is properly configured.")

    results = {}

    # Run all checks
    results['Python version'] = check_python_version()
    results['Required packages'] = check_packages()
    results['CUDA/GPU support'] = check_cuda()
    results['Model creation'] = test_model_creation()
    results['Data pipeline'] = test_data_pipeline()
    results['Training module'] = test_training_module()
    results['Evaluation module'] = test_evaluation_module()
    results['Processing modules'] = test_processing_modules()
    results['Directory structure'] = check_directory_structure()
    results['Raw data availability'] = check_raw_data_availability()
    results['Processed data availability'] = check_processed_data_availability()

    # Print summary
    print_summary(results)

    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)