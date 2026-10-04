# HybridQTN: Memory-Efficient Image Segmentation with an Axial MPS Bottleneck

Research code for **HybridQTN**, a U-Net for binary image segmentation whose 1024-channel bottleneck is replaced by a quantum-inspired, tensor-train-factorized **Axial Matrix Product State (MPS)** block. The block mixes global context along the height axis and then the width axis through matrix product operators (MPOs) over the binary digits of the joint (channel, position) index. Every channel-wide map in the bottleneck is stored in tensor-train form.

This README is written for external researchers and reviewers. Following the Docker path below reproduces every experiment end to end.

---

## 1. Architecture at a glance

| Component | Vanilla U-Net | HybridQTN |
|---|---|---|
| Encoder / decoder | depth 4, 64 → 1024 channels, bilinear upsampling | identical |
| Bottleneck (1024 × H/16 × W/16) | Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU | **TT-Conv3×3** → BN → ReLU → **Axial MPS block** |

**Tensor-train primitives** (`models/tensor_train.py`):
- Channel dimensions are quantized into small modes: 1024 = 4·4·4·4·4 and 512 = 2·4·4·4·4.
- A weight `W ∈ ℝ^{N_in × N_out}` is stored as cores `G_k ∈ ℝ^{r_{k-1} × m_k × n_k × r_k}`.
- Every bond rank is capped at the smaller of the requested rank and the unfolding dimensions, so no core carries redundant parameters.
- The dense operator is materialized from the cores on each forward pass. Training therefore uses the tensor-train parameter count while keeping the arithmetic of a dense layer.

**TT-Conv3×3 entry layer** (512 → 1024):
- A 3×3 spatial core followed by five channel cores.
- Ranks (1, 9, 72, 128, 128, 16, 1).

**Axial MPS block** (`models/mps_layer.py`):
1. `BN`
2. Separable positional encoding: row + column embeddings, C·(H + W) parameters
3. `Conv3×3 (C → D)`
4. **Row MPO**: a chain over the 10 binary digits of the joint (channel, row) index (D = 32 = 2⁵, H = 32 = 2⁵). The spatial and channel digits are interleaved, and the bond dimension χ is 64. Followed by `GELU`.
5. **Column MPO**: the same construction over (channel, column).
6. `Conv3×3 (D → C)`
7. Summed with a **TT-Conv1×1** local bypass (5 sites of 4×4, rank 32), followed by `BN → GELU`, plus a residual connection

Here D is the latent width (`--bond_dim`). Every operator's input and output shapes match the dense layers they replace, so the encoder and decoder are unchanged.

### Exact bottleneck parameter counts (512 × 512 input)

Produced by `python -m scripts.bottleneck_profiler`, which derives each count analytically and asserts it against the instantiated model:

| Bottleneck | D | Parameters | Change vs. standard |
|---|---:|---:|---:|
| Standard (two 3×3 convolutions) | — | 14,159,872 | — |
| Tensorized Axial MPS | 16 | 964,577 | −93.19 % |
| Tensorized Axial MPS | 32 | 1,292,273 | −90.87 % |
| Tensorized Axial MPS | 64 | 1,914,897 | −86.48 % |

At D = 32 the whole network has 16,081,074 parameters, against 28,948,673 for the vanilla U-Net (−44.4 %). The remaining parameters sit in the unmodified encoder and decoder stages.

---

## 2. Experimental protocol

| Aspect | Setting |
|---|---|
| Data split | 70 % train / 15 % validation / 15 % test with a **fixed split seed (42)**, independent of the training seed. CVC-ClinicDB is split by **colonoscopy sequence**, so no sequence appears in more than one partition. |
| Transforms | Training: Resize → random H/V flips → Normalize. Validation and test: **deterministic** Resize → Normalize. Stochastic transforms in an evaluation pipeline raise an error. |
| Loss | Focal Tversky (Abraham & Khan, 2019): TI = (TP + s) / (TP + w_FN·FN + w_FP·FP + s), L = (1 − TI)^{1/γ}, computed per image and averaged over the batch, with w_FN = 0.3, w_FP = 0.7, γ = 4/3, s = 1e-5. BCE + Dice is available via `--loss bce_dice`. |
| Optimisation | AdamW (lr 2·10⁻⁴, weight decay 10⁻⁴), batch 4 × 8 gradient accumulation, cosine schedule, AMP, gradient clipping at 1.0 |
| SWA | Weights are averaged over the final 25 % of epochs (SWALR, lr 5·10⁻⁵). BatchNorm statistics are recomputed at the end. Per-epoch validation always evaluates the **live** weights. |
| Model selection | Best live checkpoint by validation Dice. The SWA model is evaluated separately. |
| Reported metrics | **Test set only**, computed once after training: Dice, IoU, F2, MAE and HD95, averaged per image. HD95 is the 95th-percentile symmetric border-to-border distance in pixels at the network resolution. Keys are logged as `test_best_live_*` and `test_swa_*`. |
| Memory | `Peak_VRAM_MB` = `torch.cuda.max_memory_allocated()` per epoch, reset at the start of each epoch |
| Variance | Every configuration is run with seeds 42, 43 and 44 (`scripts/run_seeds.sh`). Report mean ± std. |

---

## 3. Reproducing with Docker (recommended)

### 3.1 Prerequisites
- Linux x86-64 host with an NVIDIA GPU.
- NVIDIA driver that supports **CUDA 13**. The pinned `torch==2.13.0` wheel ships CUDA 13 runtime libraries.
- Docker Engine and the **NVIDIA Container Toolkit**. Check the setup with:

  ```bash
  docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
  ```
- About 15 GB of free disk for the image, plus space for datasets.

### 3.2 Build the image

```bash
git clone https://github.com/egoistpizza/hybrid-qtn.git
cd hybrid-qtn
docker build -t hybrid-qtn:latest .
```

Every Python dependency is pinned in `requirements.txt`, and the base image is pinned to `python:3.12.3-slim-bookworm`.

### 3.3 Define a run helper

Datasets, checkpoints and outputs live on the host and are mounted into the container. `--shm-size` is required by the multi-worker data loaders.

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
```

Experiment tracking uses Weights & Biases:
- export `WANDB_API_KEY` to log online, or
- set `WANDB_MODE=offline` to keep runs local in the mounted `wandb/` directory, or
- set `WANDB_MODE=disabled` to turn tracking off.

### 3.4 Download the datasets

```bash
hqtn hybrid-qtn:latest scripts/download_kvasir_seg.py
hqtn hybrid-qtn:latest scripts/download_cvc_clinicdb.py
hqtn hybrid-qtn:latest scripts/download_mass_roads.py
```

| Dataset | Source | Credentials |
|---|---|---|
| Kvasir-SEG (1,000 images) | Simula | none |
| CVC-ClinicDB (612 frames, 29 sequences) | Kaggle mirror by default; `--source local --archive <file>` for a manually downloaded archive | Kaggle |
| Massachusetts Roads | Kaggle | Kaggle |

For Kaggle, export `KAGGLE_USERNAME` and `KAGGLE_KEY` before running the helper.

### 3.5 Verify the installation

```bash
hqtn hybrid-qtn:latest -m pytest tests/
hqtn hybrid-qtn:latest -m scripts.bottleneck_profiler --bond_dims 16 32 64
```

Tests that need a dataset are skipped when that dataset is not mounted.

### 3.6 Train

```bash
hqtn hybrid-qtn:latest train.py --model hybrid  --bond_dim 32 --dataset kvasir_seg --seed 42
hqtn hybrid-qtn:latest train.py --model vanilla               --dataset kvasir_seg --seed 42
```

| Argument | Values / default |
|---|---|
| `--model` | `vanilla`, `hybrid`, `deep_hybrid` |
| `--bond_dim` | default 32 |
| `--loss` | `focal_tversky` (default), `bce_dice` |
| `--fn_weight` / `--fp_weight` / `--focal_gamma` | 0.3 / 0.7 / 4/3 |
| `--dataset` | any config stem in `configs/`; comma-separated for joint training |
| `--epochs` | 100 |
| `--batch_size` | 4 |
| `--seed` | 42 |
| `--tag` | free-form run suffix |

Checkpoints are written to `checkpoints/<dataset>__<model>__<loss>__seed<seed>[__tag]/`:
- `best_model.pth`: best live checkpoint by validation Dice
- `best_swa_model.pth`: SWA model

### 3.7 Run the three seeds used in the paper

```bash
hqtn --entrypoint bash hybrid-qtn:latest scripts/run_seeds.sh --model hybrid --bond_dim 32 --dataset kvasir_seg
SEEDS="42 43 44 45 46" hqtn -e SEEDS --entrypoint bash hybrid-qtn:latest scripts/run_seeds.sh --model vanilla --dataset kvasir_seg
```

Aggregate the `test_swa_*` / `test_best_live_*` summary values across seeds as mean ± std.

### 3.8 Evaluate a trained model on the held-out test set

```bash
hqtn hybrid-qtn:latest evaluate.py --model hybrid --bond_dim 32 --dataset kvasir_seg --seed 42 --split test
```

This loads `best_swa_model.pth`, falling back to `best_model.pth` if absent. It prints Dice, IoU, MAE, F2 and HD95, and writes 5–10 qualitative panels to `outputs/visualizations/<run>/`. Use `--run <name>` to evaluate a checkpoint directory by name.

### 3.9 ONNX export and throughput

```bash
hqtn hybrid-qtn:latest export.py --model hybrid --bond_dim 32 --image_size 512 --iterations 100
```

---

## 4. Local setup without Docker

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/
python train.py --model hybrid --dataset kvasir_seg --seed 42
```

Run all commands from the repository root. Invoke the profiler as a module: `python -m scripts.bottleneck_profiler`.

---

## 5. Reproducibility notes

- `utils/seed.py` seeds Python, NumPy and PyTorch, enables `torch.use_deterministic_algorithms` (warn-only), disables cuDNN benchmarking and sets `CUBLAS_WORKSPACE_CONFIG`.
- Albumentations keeps its own RNG, so augmentations are reseeded per DataLoader worker from the seeded loader generator.
- Bit-exact GPU reproducibility is still not guaranteed: some CUDA kernels (e.g. the bilinear-upsampling backward pass) have no deterministic implementation, and `torch.compile` and mixed precision can change reduction order across GPU models. This residual variance is why every configuration is reported over multiple seeds.

---

## 6. Repository structure

```text
├── configs/              Dataset YAMLs (paths, train_transforms, eval_transforms, optional group_map)
├── dataset/              Generic YAML-driven dataset, deterministic 70/15/15 splits, worker seeding
├── models/
│   ├── unet_classic.py   Vanilla U-Net
│   ├── unet_hybrid.py    U-Net with pluggable bottleneck transforms
│   ├── mps_layer.py      Axial MPS block
│   └── factory.py        Model construction shared by train, evaluate and export
├── scripts/              Dataset downloaders, bottleneck profiler, multi-seed runner
├── utils/                Loss functions, metrics, seeding, VRAM profiling, run naming
├── tests/                Pytest suite
├── train.py              Training with live validation, SWA and test-set reporting
├── evaluate.py           Held-out evaluation and visualisation
├── export.py             ONNX export and throughput benchmark
├── Dockerfile
└── requirements.txt      Fully pinned dependencies
```

## 7. License

MIT. See `LICENSE`.
