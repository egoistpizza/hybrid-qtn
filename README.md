# HybridQTN

**Parameter-efficient global context for medical and natural image segmentation via Toeplitz operators in a U-Net bottleneck.**

HybridQTN is a U-Net for binary image segmentation whose deepest (1024-channel) stage is replaced by a tensorized block. The block mixes information across the whole feature map with two axial Toeplitz operators, one along the rows and one along the columns. Each operator weight depends only on the offset between input and output position, so the block adds global context while staying shift-equivariant. The operator kernels are stored as quantized tensor trains (QTT), and the remaining channel maps of the stage are tensor-train layers.

The tensorized stage has 1,161,409 parameters instead of 14,159,872 (−91.8 %); the full network has 15,950,210 instead of 28,948,673 (−44.9 %).

This repository contains the model, the training and evaluation pipeline, the dataset definitions and the tests used for the accompanying manuscript.

---

## Contents

1. [Method](#1-method)
2. [Results](#2-results)
3. [Installation](#3-installation)
4. [Datasets](#4-datasets)
5. [Training](#5-training)
6. [Evaluation](#6-evaluation)
7. [Parameter counts and ONNX export](#7-parameter-counts-and-onnx-export)
8. [Experimental protocol](#8-experimental-protocol)
9. [Reproducibility](#9-reproducibility)
10. [Repository structure](#10-repository-structure)
11. [License](#11-license)

---

## 1. Method

| Component | Dense U-Net | HybridQTN |
|---|---|---|
| Encoder and decoder | depth 4, 64 → 1024 channels, bilinear upsampling | identical |
| Deepest stage (1024 × H/16 × W/16) | Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU | TT-Conv3×3 → BN → ReLU → axial Toeplitz block |

**Axial Toeplitz block** (`models/mps_layer.py`), applied to a C × H × W feature map with latent width D (`--bond_dim`, default 32):

1. BatchNorm.
2. Conv3×3, C → D.
3. Row operator: `y[d, h, w] = Σ_{c, j} K[c, d, j − h] · x[c, j, w]`, followed by GELU.
4. Column operator: the same construction along each row.
5. Conv3×3, D → C.
6. Sum with a TT-Conv1×1 bypass, then BN → GELU and an identity residual.

Positions beyond the image edge contribute zero. Each kernel K covers every offset of the axis (2H slots) and is stored as a tensor train over the interleaved binary digits of the offset and of the input and output channels, with bond dimension χ = 64. All tensor-train weights are contracted to dense tensors in fp32 on every forward pass, so the arithmetic is that of the dense operator.

**Tensor-train layers** (`models/tensor_train.py`): channel dimensions are quantized into small modes (1024 = 4⁵, 512 = 2 · 4⁴), and a weight is stored as a chain of cores whose ranks are capped by the unfolding dimensions. The stage entry is a TT-Conv3×3 (512 → 1024, ranks 1, 9, 72, 128, 128, 16, 1); the bypass is a TT-Conv1×1 (rank 32).

**Constraints.** The input size is fixed at 512 × 512, and the bottleneck side must be a power of two (512 → 32 × 32).

**Variants and controls.** Every option below is selected on the command line and appears in the run name.

| Option | Run-name suffix | Description |
|---|---|---|
| `--axial_operator relative` (default) | `_rel` | Toeplitz operator with a QTT kernel |
| `--axial_operator absolute` | none | separate weights for every pair of positions (matrix product operator over the joint channel and position index) |
| `--axial_operator toeplitz_dense` | `_tdense` | Toeplitz operator with a dense kernel, no tensor train (control) |
| `--axial_operator circular` | `_circ` | offset wraps around the axis (ablation) |
| `--axial_boundary mean` | `_mean` | positions beyond the edge take the mean along the axis; 3×3 projections pad by replication (ablation) |
| `--positional_encoding` | none; `_nope` when off | learned separable positional encoding before the first projection |
| `--mixer attention` | `_attn` | multi-head self-attention over all bottleneck positions instead of the axial operators (control) |
| `--mixer none` | `_nomix` | no spatial mixing between the two projections (control) |
| `--model slim_vanilla` | | dense U-Net with a 232-channel deepest stage, matched to the tensorized network in size (control) |
| `--model deep_hybrid` | | tensorized blocks after both the 512- and the 1024-channel stages |

---

## 2. Results

Test-set results of the SWA models (focal Tversky loss, D = 32, relative operators, no positional encoding). All comparisons are per-image paired against the dense U-Net trained with the same seed on the same GPU model. Full tables, statistics and the controls are in the manuscript.

| Dataset | Metric | Dense U-Net | HybridQTN | Evidence |
|---|---|---:|---:|---|
| Kvasir-SEG | Dice | 0.799 | 0.834 | better in 12 of 12 seed pairs (four batches, three GPU models) |
| COD10K, CAMO, CHAMELEON, NC4K | S-measure | | | better on all four test sets, +0.023 to +0.071 (Holm-adjusted) |
| DUTS-TE | S-measure | 0.746 | 0.782 | better for both seeds |
| Kvasir-SEG → CVC-ClinicDB | Dice | | | better transfer for every Kvasir-SEG seed |
| CVC-ClinicDB, grouped 5-fold | Dice | 0.573 | 0.610 | numerically better, not significant at sequence level |
| Massachusetts Roads | Dice | 0.625 | 0.580 | worse; the gap comes almost entirely from road-free crops |

Controls on Kvasir-SEG (seeds 42–44) show that the gain comes from the relative global mixing: without spatial mixing, or with a dense U-Net of the same size, accuracy falls to the level of the dense U-Net. Self-attention reaches the same Dice with a larger HD95, and a dense Toeplitz kernel is about as accurate as the QTT kernel, which stores the operator with 36 % fewer parameters per axis.

---

## 3. Installation

### Docker

Requirements:
- a Linux x86-64 host with an NVIDIA GPU;
- an NVIDIA driver that supports CUDA 13, since the pinned `torch==2.13.0` wheel ships CUDA 13 runtime libraries;
- Docker Engine with the NVIDIA Container Toolkit;
- about 15 GB of disk for the image, plus space for the datasets.

```bash
git clone https://github.com/egoistpizza/hybrid-qtn.git
cd hybrid-qtn
docker build -t hybrid-qtn:latest .
```

The base image is `python:3.12.3-slim-bookworm` and every Python dependency is pinned in `requirements.txt`. The image's entry point is `python`. Datasets, checkpoints and outputs stay on the host:

```bash
mkdir -p data checkpoints outputs wandb

hqtn() {
  docker run --rm --gpus all --shm-size=8g \
    -v "$PWD/data:/workspace/hybrid-qtn/data" \
    -v "$PWD/checkpoints:/workspace/hybrid-qtn/checkpoints" \
    -v "$PWD/outputs:/workspace/hybrid-qtn/outputs" \
    -v "$PWD/wandb:/workspace/hybrid-qtn/wandb" \
    -e WANDB_API_KEY -e WANDB_MODE -e KAGGLE_USERNAME -e KAGGLE_KEY \
    "$@"
}

hqtn hybrid-qtn:latest -m pytest tests -q
```

### Local virtual environment

```bash
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python -m pytest tests -q
```

Run every command from the repository root. Tests that need a dataset are skipped when the dataset is not present under `data/`.

### Experiment tracking

Runs are logged to Weights & Biases. Export `WANDB_API_KEY` to log online, set `WANDB_MODE=offline` to keep runs in the local `wandb/` directory, or set `WANDB_MODE=disabled` to turn logging off.

---

## 4. Datasets

Each dataset is a YAML file in `configs/` that defines its sources, split policy and transforms; `--dataset <name>` selects it, and a comma-separated list trains on several datasets jointly. See `dataset/README.md` for the configuration format.

```bash
./venv/bin/python -m scripts.download_kvasir_seg --insecure
./venv/bin/python -m scripts.download_cvc_clinicdb
./venv/bin/python -m scripts.download_mass_roads
./venv/bin/python -m scripts.download_cod
./venv/bin/python -m scripts.download_duts
```

| Config | Dataset | Split | Source |
|---|---|---|---|
| `kvasir_seg` | Kvasir-SEG, 1,000 images | random 70 / 15 / 15 | Simula |
| `cvc_clinicdb` | CVC-ClinicDB, 612 frames from 29 sequences | grouped by sequence; optional grouped 5-fold cross-validation (`--fold 0`…`4`) | Kaggle mirror (credentials required), or `--source local --archive <file>` |
| `mass_roads` | Massachusetts Roads | random 70 / 15 / 15 | Kaggle (credentials required) |
| `cod` | COD10K, CAMO, CHAMELEON, NC4K; 4,040 training and 6,473 test images | official training and test sets; four named test sets | official SINet-V2 archives, non-commercial research use |
| `duts` | DUTS; 10,553 training and 5,019 test images | official DUTS-TR / DUTS-TE | saliencydetection.net |

All downloads use verified HTTPS. The Kvasir-SEG server omits an intermediate TLS certificate, hence `--insecure`; its archive is pinned by size and SHA-256 and checked in either case, as is the DUTS archive. Every downloader checks the final image and mask counts. For Kaggle sources, export `KAGGLE_USERNAME` and `KAGGLE_KEY`.

---

## 5. Training

```bash
./venv/bin/python train.py --model hybrid  --dataset kvasir_seg --seed 42
./venv/bin/python train.py --model vanilla --dataset kvasir_seg --seed 42
SEEDS="42 43 44" bash scripts/run_seeds.sh --model hybrid --dataset kvasir_seg
```

With Docker, replace `./venv/bin/python` by `hqtn hybrid-qtn:latest`; for the seed script use `hqtn -e SEEDS --entrypoint bash hybrid-qtn:latest scripts/run_seeds.sh …`.

| Argument | Default | Description |
|---|---|---|
| `--model` | `hybrid` | `vanilla`, `slim_vanilla`, `hybrid`, `deep_hybrid` |
| `--dataset` | `kvasir_seg` | config name in `configs/`; comma-separated for joint training |
| `--loss` | `focal_tversky` | `focal_tversky` or `bce_dice` |
| `--fn_weight`, `--fp_weight`, `--focal_gamma` | 0.3, 0.7, 4/3 | focal Tversky weights on false negatives and false positives, and γ (the loss exponent is 1/γ) |
| `--seed` | 42 | training seed; the data split seed is fixed at 42 |
| `--bond_dim` | 32 | latent width D of the tensorized block |
| `--axial_operator`, `--axial_boundary`, `--positional_encoding`, `--mixer` | relative, zero, off, axial | block variant (Section 1) |
| `--fold` | none | cross-validation fold for configs with `cross_validation_folds` |
| `--epochs` | 100 | |
| `--batch_size` | 4 | gradients are accumulated over 8 batches |
| `--tag` | none | suffix that separates runs of the same configuration |
| `--deterministic` | off | deterministic GPU kernels, bit-identical reruns on one GPU type, about 4× slower |

Checkpoints are written to `checkpoints/<dataset>__<architecture>__<loss>__seed<N>[__fold<k>][__<tag>]/`, for example `kvasir_seg__hybrid_unet_b32_rel_nope__focal_tversky__seed42`:
- `best_model.pth`: the live weights with the best validation Dice;
- `best_swa_model.pth`: the SWA model.

After training, both models are evaluated once on every test set of the dataset.

---

## 6. Evaluation

```bash
./venv/bin/python evaluate.py --model hybrid --dataset kvasir_seg --seed 42 --split test
```

`evaluate.py` accepts the same run options as `train.py` and resolves the same checkpoint directory; `--run <name>` selects a directory by name. It loads `best_swa_model.pth`, or `best_model.pth` if the SWA model is absent, and reports Dice, IoU, MAE, F2 and HD95 for every test set, plus S-measure, mean E-measure, weighted F-measure and map MAE for datasets that enable them (COD, DUTS). Qualitative panels are written to `outputs/visualizations/<run>/<set>/`; `--save_maps` also saves the probability maps at ground-truth resolution to `outputs/prediction_maps/<run>/<set>/`.

---

## 7. Parameter counts and ONNX export

```bash
./venv/bin/python -m scripts.bottleneck_profiler --bond_dims 32
./venv/bin/python export.py --model hybrid --bond_dim 32
```

The profiler derives every parameter count analytically and asserts it against the instantiated model; it accepts the variant options of `train.py`. Counts at D = 32 for a 512 × 512 input:

| Component of the tensorized stage | Parameters |
|---|---:|
| TT-Conv3×3 entry | 447,889 |
| Conv3×3 projections (C → D and D → C) | 590,880 |
| Row and column Toeplitz operators (QTT, 41,608 each) | 83,216 |
| TT-Conv1×1 bypass | 33,280 |
| BatchNorm layers | 6,144 |
| **Total** (dense stage: 14,159,872) | **1,161,409** |

| Network | Parameters |
|---|---:|
| Dense U-Net (`vanilla`) | 28,948,673 |
| HybridQTN, relative operators | 15,950,210 |
| HybridQTN, absolute operators | 16,015,538 |
| HybridQTN, dense Toeplitz kernel | 15,996,018 |
| HybridQTN, self-attention mixer | 15,873,330 |
| HybridQTN, no mixing | 15,866,994 |
| Size-matched dense U-Net (`slim_vanilla`) | 15,937,697 |

`export.py` writes an ONNX graph (opset 18) to `outputs/onnx/<architecture>_s512.onnx` and reports ONNX Runtime throughput. The tensor-train cores are stored in the file and contracted inside the graph.

---

## 8. Experimental protocol

| Aspect | Setting |
|---|---|
| Splits | Fixed split seed 42, independent of the training seed. CVC-ClinicDB is split by colonoscopy sequence; in 5-fold mode the 29 sequences are assigned to folds balanced by frame count and every frame is tested once. COD and DUTS use the official test sets; validation is carved from the training sources, and training images that also appear in a test source are removed. |
| Transforms | Training: resize, random flips, normalization (horizontal flips only for natural images). Evaluation: deterministic resize and normalization. |
| Loss | Focal Tversky, computed per image: TI = (TP + s) / (TP + 0.3·FN + 0.7·FP + s), L = (1 − TI)^{3/4}. BCE + Dice as an alternative. |
| Optimization | AdamW (learning rate 2·10⁻⁴, weight decay 10⁻⁴), batch 4 with 8-step gradient accumulation, cosine schedule, mixed precision, `torch.compile`, gradient clipping at 1.0. |
| SWA | Averaging over the final 25 % of epochs; SWALR ramp to 5·10⁻⁵ over 10 % of the epochs; BatchNorm statistics recomputed at the end. |
| Model selection | Every epoch validates the live weights; the best live checkpoint is selected by validation Dice. The SWA model is evaluated separately. |
| Metrics | Per-image Dice, IoU, F2, MAE and HD95 on binarized predictions at network resolution (HD95 of an empty prediction is the image diagonal). For COD and DUTS, S-measure, mean E-measure, weighted F-measure and MAE on probability maps at ground-truth resolution, matching PySODMetrics 1.6.2. Test sets are evaluated once, after training. |
| Statistics | Paired per-image bootstrap (10,000 resamples) and Wilcoxon signed-rank test; sequence-level bootstrap for CVC-ClinicDB; Holm correction across the four COD test sets. |

---

## 9. Reproducibility

- `utils/seed.py` seeds Python, NumPy and PyTorch and fixes cuDNN; Albumentations is reseeded in every DataLoader worker.
- Without `--deterministic`, some GPU kernels (notably the backward pass of bilinear upsampling) accumulate in a non-deterministic order, so reruns with the same seed differ slightly. With `--deterministic`, reruns on the same GPU type are bit-identical, at about four times the step time.
- Results differ numerically across GPU types. Each dense/tensorized comparison is therefore run with the same seed on the same GPU model.
- Run names encode the full configuration (model, variant, loss, seed, fold, tag), so a checkpoint loads only into the architecture it was trained with.

---

## 10. Repository structure

```text
├── configs/          Dataset definitions (sources, split policy, transforms)
├── dataset/          Config parsing, dataset class, random / grouped / predefined splits, loaders
├── models/
│   ├── unet.py           U-Net with optional tensorized stages
│   ├── mps_layer.py      Axial operators and the tensorized block
│   ├── tensor_train.py   Tensor-train linear and convolution layers
│   ├── variant.py        Block variants and their command-line options
│   └── factory.py        Model construction for training, evaluation and export
├── scripts/          Dataset downloaders, parameter profiler, multi-seed runner
├── utils/            Losses, metrics, evaluation loop, checkpoints, run naming and logging, seeding
├── tests/            Pytest suite
├── train.py          Training with live validation, SWA and test evaluation
├── evaluate.py       Evaluation of saved checkpoints, panels and probability maps
├── export.py         ONNX export and throughput measurement
├── Dockerfile
└── requirements.txt  Pinned dependencies
```

---

## 11. License

MIT. See `LICENSE`. The datasets are distributed under their own terms; the camouflaged-object archives are restricted to non-commercial research use.
