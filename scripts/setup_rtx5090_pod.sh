#!/usr/bin/env bash
#
# setup_rtx5090_pod.sh — provision a fresh RTX 5090 (Blackwell / sm_120) pod
# for training this repo: creates a venv, installs a driver-matched CUDA build
# of PyTorch, installs project dependencies, and verifies real GPU compute.
#
# Usage:
#   ./scripts/setup_rtx5090_pod.sh                 # venv + torch + deps + verify
#   ./scripts/setup_rtx5090_pod.sh --with-onnx     # also install ONNX export extras
#   ./scripts/setup_rtx5090_pod.sh --recreate      # delete and rebuild the venv
#   ./scripts/setup_rtx5090_pod.sh --skip-verify   # install only, no GPU checks
#
# Env overrides:
#   VENV_DIR=./venv        PYTHON_BIN=python3.12
#   CUDA_CHANNEL=cu128     TORCH_VERSION=2.11.0
#
# WHY cu128 AND NOT A NEWER BUILD:
#   PyTorch cu130 wheels need NVIDIA driver r580+. Pods commonly ship r570
#   (CUDA 12.8), where cu130 dies with "driver is too old (found version 12080)"
#   even though the GPU is fine. cu128 wheels carry native sm_120 kernels and
#   run on r570 *and* on newer drivers, so they are the safe default here.
#   Override with CUDA_CHANNEL/TORCH_VERSION only if you know your driver.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${VENV_DIR:-$REPO_ROOT/venv}"
CUDA_CHANNEL="${CUDA_CHANNEL:-cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_CHANNEL}"
MIN_DRIVER_MAJOR=570          # first driver branch with Blackwell / sm_120 support
ONNXRUNTIME_VERSION="1.26.0"  # last onnxruntime-gpu built against CUDA 12.x

WITH_ONNX=0
RECREATE=0
SKIP_VERIFY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-onnx)   WITH_ONNX=1 ;;
    --recreate)    RECREATE=1 ;;
    --skip-verify) SKIP_VERIFY=1 ;;
    --venv)        VENV_DIR="$2"; shift ;;
    -h|--help)     awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight --
log "Preflight: GPU and driver"

command -v nvidia-smi >/dev/null 2>&1 \
  || die "nvidia-smi not found. This script must run on a GPU pod with the NVIDIA driver installed."

DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d '[:space:]')"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/^ *//')"
GPU_MEM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]' || true)"
DRIVER_MAJOR="${DRIVER_VERSION%%.*}"

info "GPU          : ${GPU_NAME} (${GPU_MEM})"
info "Compute cap  : ${COMPUTE_CAP:-unknown}"
info "Driver       : ${DRIVER_VERSION}"

if [[ "$DRIVER_MAJOR" -lt "$MIN_DRIVER_MAJOR" ]]; then
  die "Driver ${DRIVER_VERSION} is older than r${MIN_DRIVER_MAJOR}, which is the first branch
       supporting Blackwell (sm_120). Ask your provider for a newer driver image."
fi

if [[ "$CUDA_CHANNEL" == "cu130" && "$DRIVER_MAJOR" -lt 580 ]]; then
  die "CUDA_CHANNEL=cu130 needs driver r580+, but this pod has ${DRIVER_VERSION}.
       Use the default cu128 instead."
fi

case "$COMPUTE_CAP" in
  12.*) : ;;                        # Blackwell, what this script targets
  "")   warn "Could not read compute capability; continuing." ;;
  *)    warn "Compute capability ${COMPUTE_CAP} is not sm_120. The cu128 build still covers
       sm_75/80/86/90/100, so this should work, but it is not the tested path." ;;
esac

# ------------------------------------------------------------------ python ---
log "Preflight: Python interpreter"

pick_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then echo "$PYTHON_BIN"; return; fi
  for c in python3.12 python3.13 python3.11 python3.10 python3 python; do
    command -v "$c" >/dev/null 2>&1 && { echo "$c"; return; }
  done
}
PY="$(pick_python)"
[[ -n "$PY" ]] || die "No python interpreter found."

PY_VER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PY_OK="$("$PY" -c 'import sys; print(1 if (3,10) <= sys.version_info[:2] <= (3,13) else 0)')"
[[ "$PY_OK" == "1" ]] \
  || die "Python ${PY_VER} is outside the range torch ${TORCH_VERSION}+${CUDA_CHANNEL} ships wheels for (3.10-3.13).
       Set PYTHON_BIN to a supported interpreter."

info "Interpreter  : $(command -v "$PY") (Python ${PY_VER})"

# -------------------------------------------------------------------- venv ---
log "Creating virtual environment: ${VENV_DIR}"

if [[ -d "$VENV_DIR" && "$RECREATE" == "1" ]]; then
  info "--recreate given, removing existing venv"
  rm -rf "$VENV_DIR"
fi

if [[ -d "$VENV_DIR" ]]; then
  info "venv already exists, reusing it (pass --recreate for a clean rebuild)"
else
  "$PY" -m venv "$VENV_DIR" 2>/dev/null || die "venv creation failed.
       On Debian/Ubuntu images you may need:  apt-get update && apt-get install -y python${PY_VER}-venv"
fi

VPY="$VENV_DIR/bin/python"
VPIP="$VENV_DIR/bin/pip"
[[ -x "$VPY" ]] || die "venv looks broken: ${VPY} is not executable"

info "Upgrading pip toolchain"
"$VPY" -m pip install --quiet --upgrade pip wheel

# ------------------------------------------------------------------- torch ---
log "Installing PyTorch ${TORCH_VERSION}+${CUDA_CHANNEL}"
info "index: ${TORCH_INDEX}"
info "This pulls ~3 GB (torch, triton, cuDNN, cuBLAS, NCCL). Give it a few minutes."

# Installed first so it, not a transitive dependency, decides the CUDA stack.
"$VPIP" install --index-url "$TORCH_INDEX" "torch==${TORCH_VERSION}+${CUDA_CHANNEL}"

# ------------------------------------------------------- project packages ----
log "Installing project dependencies"

REQ_FILTERED="$(mktemp)"
trap 'rm -f "$REQ_FILTERED"' EXIT

if [[ -f "$REPO_ROOT/requirements.txt" ]]; then
  # Strip the CUDA/torch stack: requirements.txt pins a cu130 build that cannot
  # initialise on an r570 driver. Everything else is taken as-is.
  # setuptools/wheel are filtered too: requirements.txt pins setuptools==83.0.0
  # while torch 2.11 requires setuptools<82. Letting torch's constraint win
  # avoids leaving pip in a conflicted state.
  grep -vEi '^\s*(torch|triton|nvidia-|cuda-toolkit|cuda-bindings|cuda-pathfinder|onnx|setuptools|wheel)' \
    "$REPO_ROOT/requirements.txt" > "$REQ_FILTERED" || true
  info "Installing from requirements.txt (CUDA/torch/onnx/setuptools lines filtered out)"
  "$VPIP" install -r "$REQ_FILTERED"
else
  warn "requirements.txt not found; installing a known-good baseline instead"
  "$VPIP" install albumentations opencv-python-headless numpy scipy matplotlib \
                  pyyaml tqdm wandb pytest
fi

# Imported by the code but absent from requirements.txt:
#   kaggle + rarfile -> scripts/download_cvc_clinicdb.py
#   python-dotenv    -> credential loading
info "Installing extras used by the code but missing from requirements.txt"
"$VPIP" install --quiet kaggle rarfile python-dotenv

# rarfile needs an external extractor for .rar archives; the download script
# falls back to 7z/unrar, so make sure at least one exists.
if ! command -v 7z >/dev/null 2>&1 && ! command -v unrar >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    info "Installing p7zip-full for .rar dataset extraction"
    apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq p7zip-full >/dev/null 2>&1 \
      || warn "Could not install p7zip-full. If a dataset ships as .rar, install 7z or unrar manually."
  else
    warn "No 7z/unrar found and no apt-get. .rar dataset extraction may fail."
  fi
fi

if [[ "$WITH_ONNX" == "1" ]]; then
  log "Installing ONNX export extras"
  info "Pinned to onnxruntime-gpu ${ONNXRUNTIME_VERSION}: it is the last release built"
  info "against CUDA 12.x. 1.27+ requires the CUDA 13 runtime, which would both fail"
  info "on this driver and drag a conflicting cu13 stack in beside torch's cu128."
  "$VPIP" install onnx onnxscript "onnxruntime-gpu==${ONNXRUNTIME_VERSION}"
fi

log "Checking dependency consistency"
if "$VPIP" check; then
  info "no dependency conflicts"
else
  warn "pip reports dependency conflicts (listed above). Training usually still works,
       but re-run with --recreate if you hit import errors."
fi

# --------------------------------------------------------- runtime tuning ----
log "Writing CUDA runtime environment file"

ENV_FILE="$REPO_ROOT/env.cuda.sh"
cat > "$ENV_FILE" <<ENV_EOF
# Source this before training:  source env.cuda.sh
# Generated by scripts/setup_rtx5090_pod.sh on $(date -u '+%Y-%m-%d %H:%M UTC')
#
# These are optional performance/ergonomics settings, not requirements. The
# training run works without them; they mainly help on long jobs.

source "${VENV_DIR}/bin/activate"

# Reduce allocator fragmentation on long runs with varying batch shapes.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Target Blackwell when anything compiles CUDA from source (extensions, JIT).
export TORCH_CUDA_ARCH_LIST=12.0

# Make CUDA_VISIBLE_DEVICES indices match nvidia-smi ordering.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# DataLoader uses num_workers=4; keep BLAS threads from oversubscribing the CPU.
export OMP_NUM_THREADS=4

# Uncomment to silence tokenizer/threading chatter or to pick a single GPU:
# export CUDA_VISIBLE_DEVICES=0
ENV_EOF
info "wrote ${ENV_FILE}"

# ------------------------------------------------------------------ verify ---
if [[ "$SKIP_VERIFY" == "1" ]]; then
  log "Skipping verification (--skip-verify)"
else
log "Verifying GPU compute"

"$VPY" - <<'VERIFY_EOF' || die "GPU verification failed. The environment is NOT ready."
import sys
import torch

failures = []

def check(label, fn):
    try:
        print(f"    {label:<22}: {fn()}")
    except Exception as exc:
        print(f"    {label:<22}: FAILED -> {type(exc).__name__}: {exc}")
        failures.append(label)

print(f"    {'torch':<22}: {torch.__version__}")
print(f"    {'cuda build':<22}: {torch.version.cuda}")

if not torch.cuda.is_available():
    print("\n    torch.cuda.is_available() is False -- the CUDA build does not match the driver.")
    sys.exit(1)

check("cudnn",        lambda: torch.backends.cudnn.version())
check("device",       lambda: torch.cuda.get_device_name(0))
check("capability",   lambda: "sm_%d%d" % torch.cuda.get_device_capability(0))
check("total VRAM",   lambda: "%.1f GiB" % (torch.cuda.get_device_properties(0).total_memory / 1024**3))

# Native kernels vs PTX JIT fallback: if the device arch is absent from the
# build's arch list, everything still runs but the first launch of every kernel
# pays a JIT compile.
arch_list = torch.cuda.get_arch_list()
dev_arch = "sm_%d%d" % torch.cuda.get_device_capability(0)
print(f"    {'arch list':<22}: {' '.join(arch_list)}")
if dev_arch in arch_list:
    print(f"    {'native kernels':<22}: yes ({dev_arch} compiled in)")
else:
    print(f"    {'native kernels':<22}: NO -- {dev_arch} missing, will JIT from PTX (slow first step)")

a = torch.randn(4096, 4096, device="cuda")
check("fp32 matmul", lambda: "%.4f" % float((a @ a).abs().mean()))

def autocast_dtype(dt):
    def run():
        with torch.autocast("cuda", dtype=dt):
            return str((a @ a).dtype)
    return run
check("bf16 autocast", autocast_dtype(torch.bfloat16))
check("fp16 autocast", autocast_dtype(torch.float16))

# channels_last conv + backward: the exact path train.py runs.
def conv_test():
    x = torch.randn(8, 3, 256, 256, device="cuda").to(memory_format=torch.channels_last)
    c = torch.nn.Conv2d(3, 64, 3, padding=1).cuda().to(memory_format=torch.channels_last)
    y = c(x); y.sum().backward(); torch.cuda.synchronize()
    return f"{tuple(y.shape)} channels_last={y.is_contiguous(memory_format=torch.channels_last)}"
check("cudnn conv2d", conv_test)

from torch.amp import GradScaler
check("GradScaler", lambda: GradScaler(enabled=True).is_enabled())

# train.py calls torch.compile, so Triton must be able to codegen for this arch.
def compile_test():
    import torch.nn as nn
    m = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
    m = m.cuda().to(memory_format=torch.channels_last)
    x = torch.randn(2, 3, 64, 64, device="cuda").to(memory_format=torch.channels_last)
    with torch.autocast("cuda"):
        out = torch.compile(m)(x)
    out.sum().backward(); torch.cuda.synchronize()
    import triton
    return f"ok (triton {triton.__version__})"
check("torch.compile", compile_test)

try:
    from utils import get_device
    dev = get_device()
    print(f"    {'get_device()':<22}: {dev}")
    if dev.type != "cuda":
        failures.append("get_device")
except Exception as exc:
    print(f"    {'get_device()':<22}: skipped ({exc})")

# End-to-end: the real model at the configured 512x512, forward only.
try:
    from models.unet_hybrid import UNetDeepHybrid
    from models.mps_layer import MPSBottleneck
    model = UNetDeepHybrid(
        3, 1,
        transform_512=MPSBottleneck(in_channels=512, bond_dim=32, height=64, width=64),
        transform_1024=MPSBottleneck(in_channels=1024, bond_dim=32, height=32, width=32),
    ).cuda().to(memory_format=torch.channels_last).eval()
    x = torch.randn(2, 3, 512, 512, device="cuda").to(memory_format=torch.channels_last)
    with torch.no_grad(), torch.autocast("cuda"):
        y = model(x)
    torch.cuda.synchronize()
    n = sum(p.numel() for p in model.parameters())
    print(f"    {'model forward':<22}: {tuple(y.shape)} ({n:,} params)")
    print(f"    {'peak VRAM':<22}: %.2f GiB" % (torch.cuda.max_memory_allocated() / 1024**3))
except Exception as exc:
    print(f"    {'model forward':<22}: FAILED -> {type(exc).__name__}: {exc}")
    failures.append("model forward")

if failures:
    print("\n    failed checks: " + ", ".join(failures))
    sys.exit(1)
print("\n    all GPU checks passed")
VERIFY_EOF
fi

# ------------------------------------------------------------------- lock ----
log "Freezing the working environment"
LOCK_FILE="$REPO_ROOT/requirements-${CUDA_CHANNEL}.lock.txt"
{
  echo "# Frozen by scripts/setup_rtx5090_pod.sh on $(date -u '+%Y-%m-%d %H:%M UTC')"
  echo "# GPU ${GPU_NAME} | driver ${DRIVER_VERSION} | Python ${PY_VER}"
  echo "#"
  echo "# Reinstall into a fresh venv with:"
  echo "#   pip install -r $(basename "$LOCK_FILE")"
  echo "#"
  echo "# The extra index below is required: torch ${TORCH_VERSION}+${CUDA_CHANNEL} is published"
  echo "# only by PyTorch, not by PyPI, so a plain install would fail to resolve it."
  echo "--extra-index-url ${TORCH_INDEX}"
  "$VPIP" freeze
} > "$LOCK_FILE"
info "wrote $(basename "$LOCK_FILE")"

# ---------------------------------------------------------------- summary ----
log "Done"
cat <<SUMMARY
    GPU     : ${GPU_NAME}
    Driver  : ${DRIVER_VERSION}
    Torch   : ${TORCH_VERSION}+${CUDA_CHANNEL}
    Venv    : ${VENV_DIR}

    Start training:

      source env.cuda.sh
      python train.py --model deep_hybrid --bond_dim 32

    Or without sourcing the env file:

      ${VENV_DIR}/bin/python train.py --model deep_hybrid --bond_dim 32

    WARNING: do not run 'pip install -r requirements.txt' in this venv.
    It pins torch==2.13.0 (cu130), which cannot initialise on driver
    ${DRIVER_VERSION} and will undo this setup. Re-run this script instead.
SUMMARY
