# FIDES: Fidelity-Informed Deployment for Exact Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Version: 1.0.0](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/tale-bonbom/FIDES/releases/tag/v1.0.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21421492.svg)](https://doi.org/10.5281/zenodo.21421492)

**Version 1.0.0** | [Changelog](CHANGELOG.md) | [Release v1.0.0](https://github.com/tale-bonbom/FIDES/releases/tag/v1.0.0)

Official implementation of **FIDES** (Fidelity-Informed Deployment for Exact Segmentation), grounded in the **Training-Inference Consistency Guideline (TICP)** for cardiac MRI segmentation.

> **TICP:** At deployment, every transformation applied to the input must match the corresponding training transform. Deviations create distribution shift that degrades model generalization.

## 📋 Table of Contents

- [Overview](#overview)
- [Three Golden Rules](#three-golden-rules)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Pre-trained Checkpoints](#pre-trained-checkpoints)
- [Reproducibility](#reproducibility)
- [Project Structure](#project-structure)
- [Pipeline Configurations](#pipeline-configurations)
- [Experiments](#experiments)
- [Clinical Validation](#clinical-validation)
- [Benchmarking](#benchmarking)
- [Citation](#citation)
- [License](#license)

## Overview

FIDES demonstrates that **inference pipeline fidelity** — not architectural sophistication — is the primary determinant of deployment performance in cardiac MRI segmentation. The FIDES-optimal pipeline achieves mean 3D Dice of 0.934 on ACDC, a +0.654 improvement over the broken baseline (0.280), with aspect ratio preservation alone contributing 95.7% of the total gain.

### Key Findings

| Component | ΔDice | % Contribution | TICP Principle |
|-----------|-------|----------------|----------------|
| Aspect Ratio Preservation (ARP) | +0.6256 | 95.7% | Geometric consistency |
| PSR + LBU | +0.019 | 3.0% | Geometric + Post-processing |
| eTTA-HVR | +0.0032 | 0.5% | Augmentation consistency |
| **Total** | **+0.6538** | **100%** | — |

## Three Golden Rules

1. **Preserve Geometric Fidelity** — Always maintain aspect ratio during inference
2. **Never Post-Process a Well-Trained Model** — All post-processing is neutral or harmful
3. **Only Augment With Training-Consistent Transformations** — Only discrete geometric TTA is beneficial

## Installation

```bash
git clone https://github.com/tale-bonbom/FIDES.git
cd FIDES
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA ≥ 11.8 (GPU recommended)
- See `requirements.txt` for full list

## Data Preparation

### ACDC Dataset

1. **Download** the ACDC dataset from https://www.creatis.insa-lyon.fr/Challenge/acdc/
2. **Extract** to a directory of your choice (e.g., `/data/acdc/`)
3. **Expected directory structure**:
   ```
   /data/acdc/
   ├── training/
   │   ├── patient001/
   │   │   ├── patient001_frame01.nii.gz   # ED frame
   │   │   ├── patient001_frame01_gt.nii.gz
   │   │   ├── patient001_frame12.nii.gz   # ES frame
   │   │   └── patient001_frame12_gt.nii.gz
   │   ├── patient002/
   │   └── ...
   └── testing/
       └── patientXXX/
   ```
4. **File naming**: `patientXXX_frameZZ.nii.gz` for images, `patientXXX_frameZZ_gt.nii.gz` for ground truth
5. **Labels**: 0=background, 1=LVBP (left ventricle blood pool), 2=LVM (left ventricle myocardium), 3=RV (right ventricle)

### M&Ms Dataset (Optional, for cross-dataset validation)

1. **Download** from https://www.ub.edu/mnm/
2. Extract to `/data/mms/`
3. The `MMsDataset` loader expects a similar structure; see `data/mms_loader.py` for details.

### Important: No Pre-processing Required

FIDES applies all transformations at **inference time** — there is no separate pre-processing step. The pipeline handles:
- Aspect ratio preservation (ARP) during input resizing
- Pixel spacing restoration (PSR) after prediction
- Logits-bilinear-upsampling (LBU) for smooth boundaries
- Z-score normalization is applied inside the pipeline

Simply point `--input` to the raw ACDC directory; the loader reads `.nii.gz` files directly.

## Quick Start

### Run FIDES-Optimal Pipeline

```bash
python run_fides.py --config fides_optimal --input /path/to/acdc --output ./results
```

### Run Broken Baseline (CorSeg-Original)

```bash
python run_fides.py --config corseg_original --input /path/to/acdc --output ./results
```

### Run All 38 Configurations

```bash
python run_fides.py --config all --input /path/to/acdc --output ./results
```

## Pre-trained Checkpoints

Pre-trained model checkpoints are hosted alongside the code repository:

| Model | Architecture | Training Data | Download |
|-------|---------------|---------------|----------|
| Model-A (CorSeg pre-trained) | MedNeXt-L | 1,555 subjects, 12 centers | [GitHub Release](https://github.com/tale-bonbom/FIDES/releases/tag/v1.0.0) |
| Model-B (ACDC-only, no ARP) | MedNeXt-L | 100 patients (ACDC) | [GitHub Release](https://github.com/tale-bonbom/FIDES/releases/tag/v1.0.0) |
| U-Net | U-Net | 100 patients (ACDC, ARP) | [GitHub Release](https://github.com/tale-bonbom/FIDES/releases/tag/v1.0.0) |
| BasicUNet | BasicUNet | 100 patients (ACDC, ARP) | [GitHub Release](https://github.com/tale-bonbom/FIDES/releases/tag/v1.0.0) |
| SegResNet | SegResNet | 100 patients (ACDC, ARP) | [GitHub Release](https://github.com/tale-bonbom/FIDES/releases/tag/v1.0.0) |

### Loading a Checkpoint

```bash
python run_fides.py --config fides_optimal \
    --input /path/to/acdc \
    --output ./results \
    --checkpoint /path/to/checkpoints/mednext_l.pth
```

If `--checkpoint` is not specified, the pipeline attempts to load from `./checkpoints/mednext_l.pth` by default. For reproducibility, download all checkpoints from the Release page and place them in a `./checkpoints/` directory.

## Reproducibility

### Hardware Requirements

| Configuration | Minimum | Tested |
|---------------|---------|--------|
| GPU VRAM | 6 GB | NVIDIA RTX 2060 (6 GB) |
| System RAM | 8 GB | 16 GB |
| CUDA | 11.8 | 11.8 |
| Disk Space | 2 GB (code + checkpoints) | 5 GB (with datasets) |

**Note**: All experiments in the paper were conducted on a single NVIDIA RTX 2060 (6 GB VRAM). The pipeline is compatible with any CUDA-enabled GPU with ≥6 GB VRAM.

### Random Seed

All experiments use a fixed random seed (`42`) for reproducibility. The seed is set at the beginning of each experiment script:

```python
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

To reproduce exact results, set the environment variable `CUDNN_DETERMINISTIC=1` before running.

### Environment Setup (Verified)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import fides; print('FIDES installed successfully')"
```

### Verified Platforms

| OS | Python | PyTorch | CUDA | Status |
|----|--------|---------|------|--------|
| Windows 11 | 3.9 | 2.0.1 | 11.8 | ✅ Tested |
| Ubuntu 22.04 | 3.10 | 2.1.0 | 12.1 | ✅ Compatible |
| macOS 14 | 3.9 | 2.0.1 | CPU | ✅ Compatible (slower) |

## Project Structure

```
FIDES/
├── README.md                      # This file
├── LICENSE                        # MIT License
├── requirements.txt               # Python dependencies
├── run_fides.py                   # Main entry point
│
├── fides/                         # Core pipeline library
│   ├── __init__.py
│   ├── pipeline.py                # FIDES pipeline implementation
│   ├── transforms.py              # ARP, PSR, LBU transforms
│   ├── tta.py                     # Test-time augmentation strategies
│   ├── postprocessing.py          # Post-processing strategies
│   ├── model.py                   # MedNeXt-L model wrapper
│   └── configs.py                 # 38 pipeline configurations
│
├── experiments/                   # Experiment scripts
│   ├── phase1_geometric.py        # Phase I: Geometric correction (7 configs)
│   ├── phase2_postprocessing.py   # Phase II: Post-processing (7 configs)
│   ├── phase3_augmentation.py     # Phase III: Augmentation boundary (8 configs)
│   ├── phase4_perturbation.py     # Phase IV: Input perturbation (8 configs)
│   ├── phase5_strategy.py         # Phase V: Inference strategy (8 configs)
│   ├── phase6_cross_arch.py       # Phase VI: Cross-architecture validation
│   └── phase7_cross_ablation.py   # Phase VII: Cross-ablation (2×2)
│
├── clinical/                      # Clinical measurement & statistics
│   ├── measurements.py             # EDV, ESV, EF calculation
│   ├── statistics.py              # ICC, Bland-Altman, Wilcoxon tests
│   ├── error_cancellation.py      # Theorem 1 verification
│   └── tables.py                  # Generate result tables
│
├── benchmark/                     # Performance benchmarking
│   ├── benchmark_flops.py         # FLOPs analysis
│   └── benchmark_latency.py      # Latency & memory benchmarking
│
├── data/                          # Data loading utilities
│   ├── __init__.py
│   ├── acdc_loader.py             # ACDC dataset loader
│   └── mms_loader.py              # M&Ms dataset loader
│
└── utils/                         # Utility functions
    ├── __init__.py
    ├── metrics.py                  # Dice, HD95 metrics
    └── visualization.py           # Result visualization
```

## Pipeline Configurations

The 38 configurations span five phases:

| Phase | Name | Configs | Focus |
|-------|------|---------|-------|
| I | Geometric Correction | 7 | ARP, PSR, LBU, TTA |
| II | Post-Processing | 7 | CCF, ISF, AMF, CBR, ROI |
| III | Augmentation Boundary | 8 | eTTA variants, MSE |
| IV | Input Perturbation | 8 | PCT, CLAHE, iTTA |
| V | Inference Strategy | 8 | Continuous rotations, scale |

### Key Configurations

```python
# FIDES-Optimal (recommended for Dice)
config = {
    'arp': True,       # Aspect ratio preservation
    'psr': True,       # Pixel spacing restoration
    'lbu': True,       # Logits-bilinear-upsampling
    'tta': 'hvr',      # Horizontal+Vertical flip+90° Rotation
}

# FIDES-Clinical (recommended for EF measurement)
config = {
    'arp': True,
    'psr': True,
    'lbu': True,
    'tta': None,       # No TTA (avoids EF bias)
}

# CorSeg-Original (broken baseline)
config = {
    'arp': False,      # Non-isotropic resize
    'psr': False,
    'lbu': False,      # argmax before upsampling
    'tta': None,
}
```

## Experiments

### Phase VII: Cross-Ablation (Decisive Causal Test)

```bash
python experiments/phase7_cross_ablation.py --acdc_dir /path/to/acdc
```

This 2×2 experiment cross-evaluates models trained with/without ARP against inference pipelines with/without ARP, demonstrating that consistent pipelines always outperform inconsistent ones.

### Run All Phases

```bash
# Phase I-V (38 configurations)
python experiments/phase1_geometric.py --acdc_dir /path/to/acdc
python experiments/phase2_postprocessing.py --acdc_dir /path/to/acdc
python experiments/phase3_augmentation.py --acdc_dir /path/to/acdc
python experiments/phase4_perturbation.py --acdc_dir /path/to/acdc
python experiments/phase5_strategy.py --acdc_dir /path/to/acdc

# Phase VI (cross-architecture)
python experiments/phase6_cross_arch.py --acdc_dir /path/to/acdc --mms_dir /path/to/mms

# Phase VII (cross-ablation)
python experiments/phase7_cross_ablation.py --acdc_dir /path/to/acdc
```

## Clinical Validation

```bash
# Calculate clinical measurements (EDV, ESV, EF)
python clinical/measurements.py --predictions ./results --acdc_dir /path/to/acdc

# Statistical analysis (ICC, Bland-Altman)
python clinical/statistics.py --measurements ./clinical_results

# Verify Error Cancellation Theorem (Theorem 1)
python clinical/error_cancellation.py --measurements ./clinical_results
```

## Benchmarking

```bash
# FLOPs analysis
python benchmark/benchmark_flops.py

# Latency & memory benchmarking
python benchmark/benchmark_latency.py
```

## Datasets

- **ACDC** (Automated Cardiac Diagnosis Challenge): 100 patients, 200 ED/ES frames
  - Download: https://www.creatis.insa-lyon.fr/Challenge/acdc/
- **M&Ms** (Multi-Centre, Multi-Vendor & Multi-Disease): 345 patients
  - Download: https://www.ub.edu/mnm/

## Results Summary

| Pipeline | Dice | LVBP | LVM | RV | HD95 (mm) |
|----------|------|------|-----|-----|----------|
| FIDES-Optimal | 0.934 | 0.938 | 0.909 | 0.955 | 2.44 |
| FIDES-Clinical | 0.927 | 0.932 | 0.904 | 0.949 | 2.51 |
| CorSeg-Original | 0.280 | 0.094 | 0.135 | 0.612 | 32.8 |

## Citation

If you use this code, please cite:

```bibtex
@software{bao2026fides,
  author       = {Bao, Hanyu},
  title        = {{FIDES: Fidelity-Informed Deployment for Exact Segmentation}},
  version      = {1.0.0},
  year         = {2026},
  url          = {https://github.com/tale-bonbom/FIDES},
  doi          = {10.5281/zenodo.21421492},
  license      = {MIT}
}
```

**BSPC Journal Reference Format** (for paper bibliography):

> Bao H. FIDES: Fidelity-Informed Deployment for Exact Segmentation [Software]. Version 1.0.0. GitHub: https://github.com/tale-bonbom/FIDES. Archived at Zenodo: https://doi.org/10.5281/zenodo.21421492 (2026).

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Contact

**Hanyu Bao**
Beijing University of Posts and Telecommunications
Email: bhy_tale@bupt.edu.cn
