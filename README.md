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



## Estimated Parameter Overhead (Bottleneck)
| Layer                    | Formula                   |  r=32 |  r=64 | r=128 |
| ------------------------ | ------------------------- | ----: | ----: | ----: |
| Axial MPS (Ours)         | 2 × r⁴ + 2 × C × r        | 2.17M | 33.7M |  537M |
| Vanilla bottleneck       | 2 × Conv3×3 + 2 × BN      | 14.2M |     — |     — |

*Note: The exact measurable parameter count for the Axial MPS layer at r=32 is 2,167,840, reducing the Vanilla U-Net bottleneck overhead by approximately 85% without exceeding VRAM limits.*


## Datasets

Both datasets are mirrored as Hugging Face dataset repos. The repo tree is the
same layout the loaders expect, so a download is all that is needed — nothing
under `configs/` changes.

| Dataset | Hub repo | Frames | Size | Config |
| --- | --- | ---: | ---: | --- |
| Kvasir-SEG | `<ns>/kvasir-seg` | 1000 | 53 MB | `configs/kvasir_seg.yaml` |
| CVC-ClinicDB | `<ns>/cvc-clinicdb` | 612 | 53 MB | `configs/cvc_clinicdb.yaml` |

Point the scripts at your namespace once — edit `HF_NAMESPACE` in
`scripts/hf_hub.py`, or export `HQTN_HF_ORG` — then:

```bash
python scripts/download_kvasir_seg.py
python scripts/download_cvc_clinicdb.py
```

If the mirrors are private, authenticate first with `hf auth login`.

The original sources still work as fallbacks:

```bash
python scripts/download_kvasir_seg.py --source simula
python scripts/download_cvc_clinicdb.py --source kaggle
python scripts/download_cvc_clinicdb.py --source local --archive local_zip_path.zip
```

To (re)publish the mirrors from a complete local copy:

```bash
python scripts/upload_datasets.py --dry-run
```

Drop `--dry-run` to upload. Repos are created private; `--public` opts out, and
`--include-tif` adds CVC-ClinicDB's 264 MB original TIF tree (which Pillow
cannot decode — the PNG rendering is what the loader uses).