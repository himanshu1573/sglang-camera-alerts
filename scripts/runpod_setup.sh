#!/usr/bin/env bash
# Prepare a fresh RunPod pod: this repo's client environment, SGLang, and a persistent model cache.
#
#   bash scripts/runpod_setup.sh
#   source /workspace/camalerts_env.sh
#   uv run python scripts/serve.py --variant bf16
#
# Safe to run again. Ran on a RunPod pod on 2026-10-13 (RTX PRO 4500 Blackwell, driver 580.178.04).
# The CUDA compiler step was added after that run from the commands used by hand there, and has
# not run as part of this script yet. Facts and sources are in docs/serving-notes.md.
#
# Settings (environment variables):
#   SGLANG_VERSION     SGLang release to install                  (default 0.5.21)
#   WORKSPACE          persistent volume mount                    (default /workspace)
#   SGLANG_VENV        where the SGLang environment goes          (default $WORKSPACE/venvs/sglang)
#   SGLANG_PYTHON      use this interpreter if it already has SGLang (e.g. inside the SGLang image)
#   PREFETCH_MODELS=1  also download every model in configs/models.yaml
#   ALLOW_OLD_DRIVER=1 continue even if the NVIDIA driver is older than 580

set -euo pipefail

SGLANG_VERSION="${SGLANG_VERSION:-0.5.21}"
WORKSPACE="${WORKSPACE:-/workspace}"
SGLANG_VENV="${SGLANG_VENV:-$WORKSPACE/venvs/sglang}"
ENV_FILE="${ENV_FILE:-$WORKSPACE/camalerts_env.sh}"
MIN_DRIVER_MAJOR=580
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

has_sglang() {
  "$1" -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("sglang") else 1)' \
    >/dev/null 2>&1
}

echo "==> 1/7 Checking the GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. This script must run on a pod with an NVIDIA GPU." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d ' ')"
driver_major="${driver_version%%.*}"
if [[ "$driver_major" =~ ^[0-9]+$ ]] && (( driver_major < MIN_DRIVER_MAJOR )); then
  echo "WARNING: NVIDIA driver $driver_version is older than $MIN_DRIVER_MAJOR." >&2
  echo "         SGLang $SGLANG_VERSION is built for CUDA 13, which needs driver >= $MIN_DRIVER_MAJOR." >&2
  echo "         Pick a machine with a newer driver (RunPod: Additional filters -> CUDA Versions)," >&2
  echo "         or use the last CUDA 12 image, lmsysorg/sglang:v0.5.19-cu129." >&2
  if [[ "${ALLOW_OLD_DRIVER:-0}" != "1" ]]; then
    echo "         Set ALLOW_OLD_DRIVER=1 to continue anyway." >&2
    exit 1
  fi
fi

echo "==> 2/7 Preparing $WORKSPACE"
mkdir -p "$WORKSPACE"
export HF_HOME="${HF_HOME:-$WORKSPACE/hf}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORKSPACE/.cache/uv}"
# Keep uv's own Python on the volume too: the virtual environments link to it, and the
# container disk (where it would otherwise go) is erased when the pod stops.
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$WORKSPACE/.uv-python}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
echo "    Hugging Face cache: $HF_HOME"
df -h "$WORKSPACE" | tail -n 1

echo "==> 3/7 Installing uv if it is missing"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  else
    python3 -m pip install uv
  fi
fi
uv --version

echo "==> 4/7 Installing this repo's client environment (uv sync)"
(cd "$REPO_ROOT" && uv sync)

echo "==> 5/7 Installing SGLang $SGLANG_VERSION"
if [[ -n "${SGLANG_PYTHON:-}" ]] && has_sglang "$SGLANG_PYTHON"; then
  echo "    Using SGLang from SGLANG_PYTHON=$SGLANG_PYTHON"
elif command -v python3 >/dev/null 2>&1 && has_sglang "$(command -v python3)"; then
  SGLANG_PYTHON="$(command -v python3)"
  echo "    The system python3 already has SGLang (SGLang Docker image?): $SGLANG_PYTHON"
else
  # Own environment: SGLang pins torch, transformers and openai, which must not mix with ours.
  if [[ ! -x "$SGLANG_VENV/bin/python" ]]; then
    uv venv --python 3.12 "$SGLANG_VENV"
  fi
  uv pip install --python "$SGLANG_VENV/bin/python" --prerelease=allow "sglang==$SGLANG_VERSION"
  SGLANG_PYTHON="$SGLANG_VENV/bin/python"
fi
export SGLANG_PYTHON

# FlashInfer and DeepGEMM compile their kernels on the pod when the server starts. They need an
# nvcc as new as SGLang's torch (the PyTorch 2.8.0 template ships 12.8; a Blackwell GPU needs
# 12.9+) and the `ninja` that pip installed next to the SGLang interpreter.
echo "==> CUDA compiler for SGLang's kernels"
torch_cuda="$("$SGLANG_PYTHON" -c 'import torch; print(torch.version.cuda)')"
CUDA_HOME="/usr/local/cuda-$torch_cuda"
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "    No nvcc for CUDA $torch_cuda; installing it with apt."
  apt-get update
  apt-get install -y "cuda-compiler-${torch_cuda/./-}" "cuda-libraries-dev-${torch_cuda/./-}"
fi
"$CUDA_HOME/bin/nvcc" --version | tail -n 2

echo "==> 6/7 Writing $ENV_FILE"
cat > "$ENV_FILE" <<EOF
# Written by scripts/runpod_setup.sh. Load it with: source $ENV_FILE
export PATH="\$HOME/.local/bin:\$PATH"
export HF_HOME="$HF_HOME"
export UV_CACHE_DIR="$UV_CACHE_DIR"
export UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR"
export SGLANG_PYTHON="$SGLANG_PYTHON"
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:\$PATH:$(dirname "$SGLANG_PYTHON")"
EOF
if [[ -f "$HOME/.bashrc" ]] && ! grep -qF "source $ENV_FILE" "$HOME/.bashrc"; then
  echo "source $ENV_FILE" >> "$HOME/.bashrc"
  echo "    Added 'source $ENV_FILE' to ~/.bashrc"
fi

if [[ "${PREFETCH_MODELS:-0}" == "1" ]]; then
  echo "==> Downloading the models listed in configs/models.yaml"
  models="$(cd "$REPO_ROOT" && uv run python -c '
import yaml
config = yaml.safe_load(open("configs/models.yaml"))
print("\n".join(sorted({v["hf_model"] for v in config["variants"]})))
')"
  while IFS= read -r model; do
    [[ -z "$model" ]] && continue
    echo "    $model"
    "$SGLANG_PYTHON" -c 'import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1])' "$model"
  done <<< "$models"
fi

echo "==> 7/7 Versions"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
uv --version
(cd "$REPO_ROOT" && uv run python --version)
"$SGLANG_PYTHON" -c '
import sys
from importlib.metadata import PackageNotFoundError, version
print("sglang python:", sys.executable)
for name in ("sglang", "torch", "transformers", "flashinfer-python", "xgrammar"):
    try:
        print(f"{name}: {version(name)}")
    except PackageNotFoundError:
        print(f"{name}: not installed")
'

echo
echo "Done. Next:"
echo "  source $ENV_FILE"
echo "  cd $REPO_ROOT"
echo "  uv run python scripts/serve.py --variant bf16 --dry-run"
echo "  uv run python scripts/serve.py --variant bf16"
