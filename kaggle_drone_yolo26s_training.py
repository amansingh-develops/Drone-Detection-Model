#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════════
# 🎯 DRONE DETECTION — YOLO26s PRODUCTION TRAINING NOTEBOOK
# ════════════════════════════════════════════════════════════════════════════════
#
# Model:        YOLO26s (pretrained, native end-to-end — no NMS)
# Dataset:      103,992 images (71,552 train / 16,322 val / 16,118 test)
# Classes:      drone (0), helicopter (1), airplane (2)
# Resolution:   Fixed 640×640
# Augmentation: Geometric + mild HSV only (dataset is already curated)
#
# Session Handling:
#   - Auto-detects previous training state on notebook restart
#   - Resumes from exact checkpoint (epoch, optimizer, scheduler)
#   - Safe against Kaggle 12-hour timeout / disconnection
#   - Just click "Run All" again after any interruption
#
# Design Rationale:
#   The dataset was intentionally curated with objects at ALL detection
#   ranges (micro/small/medium/large) plus 20k+ background negatives.
#   72.8% of images are already at native 640px. The augmentation
#   strategy RESPECTS this curation — no mosaic (would shrink intentional
#   small objects below threshold), no artificial scale changes, no
#   copy-paste. Only real-world variation: orientation flips, mild
#   camera tilt, and lighting/color shifts.
#
# ════════════════════════════════════════════════════════════════════════════════


# %% [markdown]
# # 🎯 Drone Detection — YOLO26s Training
# **Session-resilient training notebook for Kaggle**
#
# Just click **Run All**. If the session disconnects, click **Run All** again —
# training resumes automatically from the last saved checkpoint.


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: ENVIRONMENT SETUP
# ══════════════════════════════════════════════════════════════════════════════

# %%
import subprocess
import sys

def install_packages():
    """Install required packages if not already present."""
    packages = ['ultralytics', 'sahi']
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg} already installed")
        except ImportError:
            print(f"  ⏳ Installing {pkg}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])
            print(f"  ✓ {pkg} installed")

print("Setting up environment...")
install_packages()

import os
import json
import time
import yaml
import torch
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

import ultralytics
from ultralytics import YOLO

print(f"\n{'─'*50}")
print(f"  Ultralytics : {ultralytics.__version__}")
print(f"  PyTorch     : {torch.__version__}")
print(f"  CUDA        : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE'}")
if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f"  VRAM        : {vram_gb:.1f} GB")
print(f"  Python      : {sys.version.split()[0]}")
print(f"{'─'*50}")

if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU detected. Enable GPU acceleration:\n"
        "  Kaggle: Settings → Accelerator → GPU P100 or T4×2\n"
        "  Colab: Runtime → Change runtime type → T4 GPU"
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
# All settings in one place. Modify here only — nothing hardcoded elsewhere.

# %%

# ─── Paths ───────────────────────────────────────────────────────────────────
# Auto-detect dataset location (works with any Kaggle dataset name)
KAGGLE_INPUT = Path('/kaggle/input')
WORKING_DIR = Path('/kaggle/working')
RUNS_DIR = WORKING_DIR / 'drone_training'

# Search for our dataset in Kaggle input
DATASET_PATH = None
if KAGGLE_INPUT.exists():
    for candidate in KAGGLE_INPUT.iterdir():
        if (candidate / 'images' / 'train').exists():
            DATASET_PATH = candidate
            break
        # Also check if dataset.yaml exists at root
        if (candidate / 'dataset.yaml').exists():
            DATASET_PATH = candidate
            break

if DATASET_PATH is None:
    # Fallback: try local path (for testing outside Kaggle)
    local_paths = [
        Path('/home/aman-singh-bhadoriya/DroneDetectionDataset'),
        Path('/run/media/aman-singh-bhadoriya/Geonix PowerShell S3 - PHDD/DroneDetectionModel/CuratedDataset'),
        Path('./CuratedDataset'),
    ]
    for lp in local_paths:
        if lp.exists() and (lp / 'images' / 'train').exists():
            DATASET_PATH = lp
            WORKING_DIR = Path('./working')
            RUNS_DIR = WORKING_DIR / 'drone_training'
            break

if DATASET_PATH is None:
    raise FileNotFoundError(
        "Dataset not found! Ensure the Kaggle Dataset is attached to this notebook.\n"
        "Go to: Add Data → Your Datasets → select 'drone-detection-curated'"
    )

print(f"📂 Dataset found: {DATASET_PATH}")

# ─── Model ───────────────────────────────────────────────────────────────────
MODEL_NAME = 'yolo26s.pt'    # YOLO26s — native end-to-end, no NMS overhead

# ─── Training Hyperparameters ────────────────────────────────────────────────
CONFIG = {
    # Core training
    'epochs': 100,
    'imgsz': 640,              # Fixed 640 — matches 72.8% of dataset natively
    'batch': -1,               # AutoBatch — finds optimal batch for your GPU
    'patience': 50,            # Early stopping if no improvement for 50 epochs
    'optimizer': 'SGD',        # Stable convergence for large datasets
    'lr0': 0.01,               # Initial learning rate
    'lrf': 0.01,               # Final LR = lr0 × lrf = 0.0001 (cosine decay)
    'momentum': 0.937,         # SGD momentum
    'weight_decay': 0.0005,    # L2 regularization
    'cos_lr': True,            # Cosine annealing schedule

    # Warmup (prevents initial gradient shock to pretrained backbone)
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,

    # ─── Augmentation ────────────────────────────────────────────────────
    # RATIONALE: The dataset was curated with ALL detection ranges
    # (micro → small → medium → large) intentionally represented.
    # Augmentation must RESPECT this curation, not override it.
    #
    # DISABLED — would damage the curated scale distribution:
    'mosaic': 0.0,             # OFF: would shrink 22.5% small objects below threshold
    'close_mosaic': 0,         # N/A (mosaic disabled)
    'scale': 0.0,              # OFF: scale diversity already exists in dataset
    'copy_paste': 0.0,         # OFF: 71k images is sufficient, no synthetic needed
    'mixup': 0.0,              # OFF: blending images adds noise without benefit
    #
    # ENABLED — adds real-world variation without changing object scales:
    'fliplr': 0.5,             # Horizontal flip: targets approach from any direction
    'flipud': 0.3,             # Vertical flip: valid for aerial/ground camera views
    'degrees': 5.0,            # Mild rotation: simulates camera tilt
    'translate': 0.1,          # Slight shift: objects aren't always centered
    'perspective': 0.0,        # No perspective warp (unnecessary)
    'erasing': 0.3,            # Random erasing: teaches partial-occlusion robustness
    #
    # Color/brightness — mild, to handle deployment lighting variation:
    'hsv_h': 0.01,             # Very light hue: sunrise/sunset color shifts
    'hsv_s': 0.15,             # Moderate saturation: safe for 4% thermal images
    'hsv_v': 0.25,             # Brightness: dawn/dusk/overcast variation

    # ─── Performance ─────────────────────────────────────────────────────
    'amp': True,               # Mixed precision — 2× faster, no accuracy loss
    'workers': 4,              # DataLoader workers (Kaggle has 4 CPU cores)
    'cache': False,            # Dataset too large for RAM cache on Kaggle
    'save_period': 5,          # Save checkpoint every 5 epochs
    'val': True,               # Validate after each epoch
    'plots': True,             # Generate training curves and confusion matrix
    'verbose': True,

    # ─── Output ──────────────────────────────────────────────────────────
    'project': str(RUNS_DIR),
    'name': 'yolo26s_drone',
    'exist_ok': True,          # Reuse same output directory (required for resume)
}

# Print config summary
print(f"\n{'═'*50}")
print(f"  TRAINING CONFIGURATION")
print(f"{'═'*50}")
print(f"  Model     : {MODEL_NAME}")
print(f"  Epochs    : {CONFIG['epochs']}")
print(f"  Image Size: {CONFIG['imgsz']}×{CONFIG['imgsz']}")
print(f"  Optimizer : {CONFIG['optimizer']} (lr0={CONFIG['lr0']}, cos_lr={CONFIG['cos_lr']})")
print(f"  Mosaic    : {'DISABLED' if CONFIG['mosaic'] == 0 else CONFIG['mosaic']}")
print(f"  Scale     : {'DISABLED' if CONFIG['scale'] == 0 else CONFIG['scale']}")
print(f"  AMP       : {CONFIG['amp']}")
print(f"  AutoBatch : {CONFIG['batch'] == -1}")
print(f"{'═'*50}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: DATASET SETUP & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

# %%
def create_dataset_yaml():
    """Create dataset.yaml pointing to the correct Kaggle paths."""
    yaml_content = {
        'path': str(DATASET_PATH),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 3,
        'names': ['drone', 'helicopter', 'airplane'],
    }

    yaml_path = WORKING_DIR / 'dataset.yaml'
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Created: {yaml_path}")
    return str(yaml_path)


def validate_dataset():
    """Quick integrity check — verify files exist and labels are valid."""
    print(f"\n{'─'*50}")
    print("DATASET VALIDATION")
    print(f"{'─'*50}")

    all_good = True

    for split in ['train', 'val', 'test']:
        img_dir = DATASET_PATH / 'images' / split
        lbl_dir = DATASET_PATH / 'labels' / split

        img_count = len(list(img_dir.glob('*')))
        lbl_count = len(list(lbl_dir.glob('*')))

        status = "✓" if img_count > 0 and img_count == lbl_count else "✗"
        if status == "✗":
            all_good = False

        print(f"  {status} {split:6s}: {img_count:,} images, {lbl_count:,} labels")

    # Quick class distribution check on a sample
    print(f"\n  Class distribution (sampled from train):")
    cls_counter = Counter()
    total_boxes = 0
    empty_labels = 0
    train_lbls = sorted((DATASET_PATH / 'labels' / 'train').glob('*.txt'))

    for lf in train_lbls[:10000]:  # Sample 10k labels for speed
        content = lf.read_text().strip()
        if not content:
            empty_labels += 1
            continue
        for line in content.split('\n'):
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(float(parts[0]))
                cls_counter[cls_id] += 1
                total_boxes += 1

    class_names = {0: 'drone', 1: 'helicopter', 2: 'airplane'}
    for cls_id in sorted(cls_counter.keys()):
        name = class_names.get(cls_id, f'class_{cls_id}')
        count = cls_counter[cls_id]
        pct = count / total_boxes * 100 if total_boxes > 0 else 0
        print(f"    {cls_id}: {name:12s} → {count:,} boxes ({pct:.1f}%)")
    print(f"    Background negatives: {empty_labels:,} (in 10k sample)")

    if not all_good:
        raise ValueError("Dataset validation failed! Check file counts above.")

    print(f"\n  ✓ Dataset validation passed")
    return True


# Create dataset YAML and validate
DATASET_YAML = create_dataset_yaml()
validate_dataset()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: SMART SESSION RESUME
# ══════════════════════════════════════════════════════════════════════════════
# Handles three scenarios:
#   1. FRESH_START  — No previous training exists → start from pretrained
#   2. RESUME       — Session crashed mid-training → resume from last.pt
#   3. COMPLETE     — All epochs finished → skip to evaluation
#
# Just run the notebook again after any interruption. It figures out what to do.

# %%
def detect_training_state():
    """
    Detect current training state by inspecting checkpoint files.

    Returns:
        tuple: (state_name, checkpoint_path, completed_epochs)
            state_name: 'FRESH_START', 'RESUME', or 'COMPLETE'
            checkpoint_path: Path to checkpoint file (or None)
            completed_epochs: Number of completed epochs (or 0)
    """
    train_dir = RUNS_DIR / 'yolo26s_drone'
    last_pt = train_dir / 'weights' / 'last.pt'
    best_pt = train_dir / 'weights' / 'best.pt'

    if not last_pt.exists():
        return 'FRESH_START', None, 0

    # Load checkpoint to inspect training progress
    try:
        ckpt = torch.load(str(last_pt), map_location='cpu', weights_only=False)

        # Extract epoch info from checkpoint
        completed_epoch = ckpt.get('epoch', -1)
        if completed_epoch == -1:
            # Try alternative key names
            train_args = ckpt.get('train_args', {})
            completed_epoch = train_args.get('epoch', -1)

        total_epochs = CONFIG['epochs']

        # Ultralytics stores epoch as 0-indexed, and -1 means training completed
        if completed_epoch == -1 or completed_epoch >= total_epochs - 1:
            return 'COMPLETE', best_pt if best_pt.exists() else last_pt, total_epochs

        return 'RESUME', last_pt, completed_epoch + 1

    except Exception as e:
        print(f"  ⚠️ Could not read checkpoint: {e}")
        print(f"  Attempting resume anyway...")
        return 'RESUME', last_pt, -1


# Detect state
STATE, CHECKPOINT, COMPLETED_EPOCHS = detect_training_state()

print(f"\n{'═'*50}")
print(f"  SESSION STATE DETECTION")
print(f"{'═'*50}")

if STATE == 'FRESH_START':
    print(f"  State  : 🟢 FRESH START")
    print(f"  Action : Load pretrained {MODEL_NAME} and begin training")
    print(f"  Epochs : 0 / {CONFIG['epochs']}")

elif STATE == 'RESUME':
    remaining = CONFIG['epochs'] - COMPLETED_EPOCHS if COMPLETED_EPOCHS > 0 else '?'
    print(f"  State  : 🟡 RESUME FROM CHECKPOINT")
    print(f"  Action : Resume training from epoch {COMPLETED_EPOCHS}")
    print(f"  Epochs : {COMPLETED_EPOCHS} / {CONFIG['epochs']} ({remaining} remaining)")
    print(f"  Checkpoint: {CHECKPOINT}")

elif STATE == 'COMPLETE':
    print(f"  State  : ✅ TRAINING COMPLETE")
    print(f"  Action : Skip to evaluation and export")
    print(f"  Best model: {CHECKPOINT}")

print(f"{'═'*50}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: TRAINING EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

# %%
def run_training():
    """Execute training with session-resilient resume logic."""

    if STATE == 'COMPLETE':
        print("\n✅ Training already complete. Skipping to evaluation.")
        best_pt = RUNS_DIR / 'yolo26s_drone' / 'weights' / 'best.pt'
        return str(best_pt)

    if STATE == 'RESUME':
        print(f"\n🔄 Resuming training from epoch {COMPLETED_EPOCHS}...")
        print(f"   Loading checkpoint: {CHECKPOINT}")
        model = YOLO(str(CHECKPOINT))
        results = model.train(resume=True)

    else:  # FRESH_START
        print(f"\n🚀 Starting fresh training with {MODEL_NAME}...")
        model = YOLO(MODEL_NAME)

        # Log training start
        start_info = {
            'start_time': datetime.now().isoformat(),
            'model': MODEL_NAME,
            'dataset': str(DATASET_PATH),
            'config': {k: v for k, v in CONFIG.items()
                       if k not in ('project', 'name', 'exist_ok')},
        }
        info_path = WORKING_DIR / 'training_info.json'
        with open(info_path, 'w') as f:
            json.dump(start_info, f, indent=2)

        results = model.train(data=DATASET_YAML, **CONFIG)

    # Training completed (or resumed and completed)
    best_pt = RUNS_DIR / 'yolo26s_drone' / 'weights' / 'best.pt'
    last_pt = RUNS_DIR / 'yolo26s_drone' / 'weights' / 'last.pt'

    print(f"\n{'═'*50}")
    print(f"  ✅ TRAINING COMPLETE")
    print(f"{'═'*50}")
    print(f"  Best model : {best_pt}")
    print(f"  Last model : {last_pt}")

    if best_pt.exists():
        size_mb = best_pt.stat().st_size / (1024**2)
        print(f"  Model size : {size_mb:.1f} MB")

    return str(best_pt)


# Execute training
training_start_time = time.time()
BEST_MODEL_PATH = run_training()
training_elapsed = time.time() - training_start_time

if training_elapsed > 60:
    print(f"\n  ⏱️ Training time: {training_elapsed/3600:.1f} hours")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: EVALUATION ON TEST SET
# ══════════════════════════════════════════════════════════════════════════════
# Runs the best model on the held-out TEST split for unbiased performance metrics.

# %%
def evaluate_model(model_path):
    """Run comprehensive evaluation on the test set."""
    print(f"\n{'═'*50}")
    print(f"  EVALUATION — TEST SET")
    print(f"{'═'*50}")

    if not Path(model_path).exists():
        print(f"  ✗ Model not found: {model_path}")
        return None

    model = YOLO(model_path)

    # Run validation on test split
    print(f"\n  Running inference on test set...")
    results = model.val(
        data=DATASET_YAML,
        split='test',
        imgsz=640,
        batch=-1,
        conf=0.001,         # Low conf threshold for comprehensive PR curves
        iou=0.6,            # IoU threshold for mAP calculation
        plots=True,
        save_json=True,
        project=str(RUNS_DIR),
        name='test_evaluation',
        exist_ok=True,
    )

    # Extract and display metrics
    print(f"\n{'─'*50}")
    print(f"  TEST SET RESULTS")
    print(f"{'─'*50}")

    # Overall metrics
    mp = results.results_dict.get('metrics/precision(B)', 0)
    mr = results.results_dict.get('metrics/recall(B)', 0)
    map50 = results.results_dict.get('metrics/mAP50(B)', 0)
    map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 0)

    print(f"\n  Overall Performance:")
    print(f"    Precision    : {mp:.4f}")
    print(f"    Recall       : {mr:.4f}")
    print(f"    mAP@50       : {map50:.4f}")
    print(f"    mAP@50-95    : {map50_95:.4f}")

    # Per-class metrics (if available)
    class_names = ['drone', 'helicopter', 'airplane']
    if hasattr(results, 'box') and hasattr(results.box, 'maps'):
        maps_per_class = results.box.maps  # mAP50-95 per class
        ap50_per_class = results.box.ap50  # AP50 per class

        print(f"\n  Per-Class Breakdown:")
        print(f"  {'Class':15s} {'AP@50':>10s} {'AP@50-95':>10s}")
        print(f"  {'─'*37}")
        for i, name in enumerate(class_names):
            if i < len(maps_per_class):
                print(f"  {name:15s} {ap50_per_class[i]:10.4f} {maps_per_class[i]:10.4f}")

    print(f"\n  Evaluation plots saved to: {RUNS_DIR / 'test_evaluation'}")

    # Save results summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model': model_path,
        'test_images': len(list((DATASET_PATH / 'images' / 'test').glob('*'))),
        'precision': float(mp),
        'recall': float(mr),
        'mAP50': float(map50),
        'mAP50_95': float(map50_95),
    }

    summary_path = WORKING_DIR / 'test_results.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✓ Results saved: {summary_path}")

    return results


# Run evaluation
eval_results = evaluate_model(BEST_MODEL_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: SAHI SLICED INFERENCE EVALUATION (OPTIONAL)
# ══════════════════════════════════════════════════════════════════════════════
# Compares standard inference vs SAHI sliced inference on the test set.
# SAHI is the recommended inference mode for 1080p/4K deployment feeds
# where distant drones appear as <32px targets.

# %%
def evaluate_with_sahi(model_path, num_test_images=500):
    """
    Run SAHI sliced inference on a subset of test images and compare
    detection counts with standard inference.

    This demonstrates the value of SAHI for deployment on high-res feeds.
    Full SAHI evaluation on all test images is slow — we sample for speed.
    """
    print(f"\n{'═'*50}")
    print(f"  SAHI SLICED INFERENCE COMPARISON")
    print(f"{'═'*50}")

    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction, get_prediction
    except ImportError:
        print("  ⚠️ SAHI not installed. Skipping sliced inference evaluation.")
        return

    if not Path(model_path).exists():
        print(f"  ✗ Model not found: {model_path}")
        return

    # Get test images (sample for speed)
    test_imgs = sorted((DATASET_PATH / 'images' / 'test').glob('*'))
    if len(test_imgs) > num_test_images:
        step = len(test_imgs) // num_test_images
        test_imgs = test_imgs[::step][:num_test_images]

    print(f"  Testing on {len(test_imgs)} images...")

    # Load model for SAHI
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=model_path,
        confidence_threshold=0.25,
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
    )

    standard_detections = 0
    sahi_detections = 0

    for i, img_path in enumerate(test_imgs):
        # Standard inference
        result_standard = get_prediction(
            str(img_path),
            detection_model,
            image_size=640,
        )
        standard_detections += len(result_standard.object_prediction_list)

        # SAHI sliced inference (640×640 window, 20% overlap)
        result_sahi = get_sliced_prediction(
            str(img_path),
            detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )
        sahi_detections += len(result_sahi.object_prediction_list)

        if (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{len(test_imgs)}...")

    print(f"\n  Results ({len(test_imgs)} images):")
    print(f"    Standard inference : {standard_detections} detections")
    print(f"    SAHI sliced        : {sahi_detections} detections")
    improvement = ((sahi_detections - standard_detections) / max(standard_detections, 1)) * 100
    print(f"    Detection increase : {improvement:+.1f}%")

    if improvement > 5:
        print(f"\n  💡 SAHI recovers {improvement:.0f}% more detections — recommended")
        print(f"     for deployment on 1080p/4K camera feeds.")
    else:
        print(f"\n  ℹ️ SAHI provides minimal improvement on this test set.")
        print(f"     Dataset images are mostly 640px native — SAHI benefits")
        print(f"     appear primarily on higher-resolution deployment feeds.")


# Run SAHI comparison (on subset for speed)
evaluate_with_sahi(BEST_MODEL_PATH, num_test_images=300)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: MODEL EXPORT
# ══════════════════════════════════════════════════════════════════════════════
# Export the best model for deployment.

# %%
def export_model(model_path):
    """Export the trained model to deployment formats."""
    print(f"\n{'═'*50}")
    print(f"  MODEL EXPORT")
    print(f"{'═'*50}")

    if not Path(model_path).exists():
        print(f"  ✗ Model not found: {model_path}")
        return

    model = YOLO(model_path)

    # Export to ONNX (universal deployment format)
    print(f"\n  Exporting to ONNX...")
    try:
        onnx_path = model.export(format='onnx', imgsz=640, simplify=True)
        print(f"  ✓ ONNX: {onnx_path}")
    except Exception as e:
        print(f"  ⚠️ ONNX export failed: {e}")

    # Export to TorchScript (PyTorch native deployment)
    print(f"\n  Exporting to TorchScript...")
    try:
        ts_path = model.export(format='torchscript', imgsz=640)
        print(f"  ✓ TorchScript: {ts_path}")
    except Exception as e:
        print(f"  ⚠️ TorchScript export failed: {e}")

    # Copy best model to working directory for easy download
    final_model = WORKING_DIR / 'best_yolo26s_drone.pt'
    shutil.copy2(model_path, str(final_model))
    print(f"\n  ✓ Best model copied to: {final_model}")

    # List all output artifacts
    print(f"\n  📦 Output Artifacts:")
    for f in sorted(WORKING_DIR.glob('*')):
        if f.is_file():
            size = f.stat().st_size
            if size > 1024*1024:
                print(f"    {f.name:40s} {size/(1024**2):.1f} MB")
            else:
                print(f"    {f.name:40s} {size/1024:.1f} KB")


export_model(BEST_MODEL_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: TRAINING SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

# %%
def print_summary():
    """Print final training summary."""
    print(f"\n{'═'*60}")
    print(f"  🎯 DRONE DETECTION TRAINING — COMPLETE")
    print(f"{'═'*60}")
    print(f"""
  Model       : YOLO26s
  Dataset     : 71,552 train / 16,322 val / 16,118 test
  Classes     : drone, helicopter, airplane
  Resolution  : 640×640 (fixed)
  Augmentation: Geometric + mild HSV (respecting curated distribution)

  Artifacts saved to: {WORKING_DIR}
  Training plots at : {RUNS_DIR / 'yolo26s_drone'}

  Next Steps:
  ─────────────────────────────────────────────────────────
  1. Download best_yolo26s_drone.pt from the Output tab
  2. For 1080p/4K deployment feeds, use SAHI sliced inference
  3. For real-time edge deployment, use direct 640 inference
  4. If helicopter/airplane accuracy is low, consider:
     - Light oversampling (2×) of rare class images
     - Re-training with class-weighted loss
    """)

    # Print results if available
    results_path = WORKING_DIR / 'test_results.json'
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print(f"  Test Set Performance:")
        print(f"    Precision : {results['precision']:.4f}")
        print(f"    Recall    : {results['recall']:.4f}")
        print(f"    mAP@50    : {results['mAP50']:.4f}")
        print(f"    mAP@50-95 : {results['mAP50_95']:.4f}")
    print(f"{'═'*60}")


print_summary()
