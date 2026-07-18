# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-18

### Added
- Initial public release of the FIDES inference pipeline
- Full implementation of the Training-Inference Consistency Guideline (TICP)
- 38 pipeline configurations across 5 phases (geometric, post-processing, augmentation, perturbation, strategy)
- Phase VI cross-architecture validation (MedNeXt-L, U-Net, BasicUNet, SegResNet)
- Phase VII 2×2 cross-ablation experiment (decisive causal evidence)
- Clinical measurement utilities (EDV, ESV, EF calculation)
- Statistical analysis (ICC(2,1), Bland-Altman, Wilcoxon signed-rank tests)
- Error Cancellation Theorem (Theorem 1) verification code
- FLOPs and latency benchmarking scripts
- ACDC and M&Ms dataset loaders
- Comprehensive README with Data Preparation and Reproducibility sections
- MIT License
- Fixed random seed (42) for reproducibility

### Verified
- Reproducibility on NVIDIA RTX 2060 (6 GB VRAM), CUDA 11.8
- Compatibility with Python 3.9-3.10 and PyTorch 2.0-2.1
- All 38 configurations produce results consistent with the paper

### Reference
This release corresponds to the paper submitted to Biomedical Signal Processing
and Control: "FIDES: When Inference Fidelity Outweighs Architecture — A
Training-Inference Consistency Guideline for Cardiac MRI Segmentation" by
Hanyu Bao (Beijing University of Posts and Telecommunications, 2026).

## [Unreleased]

### Planned
- PyTorch Lightning integration for training scripts
- Automated unit tests for pipeline components
- Docker container for environment reproducibility
- HuggingFace Hub integration for model hosting
