# Hybrid QTN: Memory-Efficient Image Segmentation using Matrix Product States

This repository contains the research codebase for the Hybrid QTN project. 

## Overview
Dense image segmentation models like U-Net face significant memory constraints due to high-dimensional matrix multiplications in their deepest layers. This bottleneck limits their scalability on standard hardware.

We address this by integrating a Quantum-Inspired Tensor Network layer. By replacing the standard U-Net bottleneck with **Matrix Product States (MPS)**, we aim to substantially reduce the parameter count and VRAM usage while preserving the spatial correlations required for medical image segmentation. The architecture is benchmarked on the MedMNIST dataset.

## Core Methodology
* **Architecture Modification:** We remove the dense bottleneck layers of a standard U-Net and introduce a custom PyTorch-based MPS layer.
* **Tensor Decomposition:** Instead of computing large dense weight matrices, the MPS layer uses SVD to decompose high-rank tensors into a sequence of low-rank tensors.
* **Feature Preservation:** Truncating insignificant singular values compresses the parameter space, while the tensor network structure preserves the latent semantic features required for the decoder.

## Repository Structure
Please adhere to the following structure to maintain reproducibility.

```text
├── data/                  # Data loaders and preprocessing (MedMNIST)
├── models/                
│   ├── unet_classic.py    # Baseline U-Net architecture
│   ├── mps_layer.py       # Custom PyTorch MPS layer
│   └── unet_hybrid.py     # Integrated Hybrid U-Net
├── utils/                 
│   ├── metrics.py         # Evaluation metrics (Dice Score, IoU)
│   ├── hardware.py        # Profiling tools (Peak VRAM, latency)
│   └── seed.py            # Global random seed fixers
├── configs/               # .yaml files for experiment hyperparameters
├── paper/                 # LaTeX source files for the manuscript
├── train.py               # Main training loop
├── evaluate.py            # Inference and testing script
└── requirements.txt       # Project dependencies
```

## Installation
Using a virtual environment is recommended to avoid dependency conflicts.

```bash
git clone https://github.com/egoistpizza/hybrid-qtn.git
cd hybrid-qtn

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Team Workflow
To ensure the stability of the codebase:

* No direct commits to `main`.
* **Branching:** Create a descriptive branch for your work (e.g., `feature/mps-layer`, `fix/dataloader`).
* **Pull Requests:** When your feature is ready, open a PR against `main` for review.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
