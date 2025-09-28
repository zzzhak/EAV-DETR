# EAV-DETR: Efficient Arbitrary-View Oriented Object Detection with Probabilistic Guarantees for UAV Imagery

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/zzzhak/EAV-DETR.git
cd EAV-DETR

# Create conda environment
conda create -n eav-detr python=3.8
conda activate eav-detr

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

#### CODrone Dataset

First, download and organize your original CODrone dataset:
```bash
# Original dataset structure
dataset/
├── CODrone/
│   ├── train/
│   │   ├── images/          # Original training images
│   │   └── labelTxt/        # DOTA format annotations
│   ├── val/
│   │   ├── images/          # Original validation images
│   │   └── labelTxt/        # DOTA format annotations
│   └── test/                # Optional test set
│       ├── images/
│       └── labelTxt/
```

Then, run the data preprocessing script to split large images into 1024×1024 patches:

```bash
cd AerialDetection/DOTA_devkit

# Run data preparation
python prepare_dota1.py --srcpath ./dataset/CODrone --dstpath ./dataset/CODrone1024

# This will generate:
dataset/
├── CODrone1024/
│   ├── train1024/
│   │   ├── images/          # Split training images (1024×1024)
│   │   ├── labelTxt/        # Split annotations
│   │   └── CODrone_train1024.json  # COCO format
│   ├── val1024/
│   │   ├── images/          # Split validation images
│   │   ├── labelTxt/        # Split annotations
│   │   └── CODrone_val1024.json    # COCO format
│   └── test1024/            # If test set exists
│       ├── images/
│       ├── labelTxt/
│       └── CODrone_test1024.json
```

#### UAV-ROD Dataset
```bash
# Download and organize UAV-ROD dataset  
dataset/
├── UAV-ROD/
│   ├── train/
│   │   ├── images/
│   │   └── annotations/
│   └── test/
│       ├── images/
│       └── annotations/
```

### Training

```bash
# Train on CODrone dataset
python tools/train.py --config configs/rtdetr/r50vd_codrone.yml --seed 42
# Train on UAV-ROD dataset  
python tools/train.py --config configs/rtdetr/r50vd_uav_rod.yml --seed 42
```

### Model Benchmarking

```bash
python tools/benchmark.py --config configs/rtdetr/r50vd_codrone.yml --checkpoint outputs/codrone/best_model.pth --batch-size 1
```

### Conformal Prediction

```bash
# Calibration
python tools/calibrate.py --config configs/rtdetr/r50vd_codrone.yml --checkpoint outputs/codrone/best_model.pth --output conformal_params.json --alpha 0.1

# Inference
python tools/inference_with_cp.py --config configs/rtdetr/r50vd_codrone.yml --checkpoint outputs/codrone/best_model.pth --conformal-params conformal_params.json --alpha 0.1
```

## 🙏 Acknowledgements

- [RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- [ai4rs](https://github.com/wokaikaixinxin/ai4rs)
- [AerialDetection](https://github.com/dingjiansw101/AerialDetection)
- [CODrone Dataset](https://github.com/AHideoKuzeA/CODrone-A-Comprehensive-Oriented-Object-Detection-benchmark-for-UAV) 
- [UAV-ROD Dataset](https://github.com/fengkaibit/UAV-ROD)
---

<div align="center">
⭐ <b>If you find this project helpful, please give us a star!</b> ⭐
</div>
