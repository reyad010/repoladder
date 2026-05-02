#!/bin/bash
# Idempotently bootstrap the two virtualenvs the artifacts need:
#   - $ROOT_DIR/my_venv        -- CPU torch wheel  (tee_server.py side)
#   - $ROOT_DIR/my_venv_cuda   -- CUDA torch wheel (backbone.py / search.py side)
#
# Both venvs install requirements.txt on top.
# Sourced by artifact1_nas/run.sh and artifact2_e2e/run.sh.
# Exports:
#   CPU_VENV, CPU_PYTHON   (bin/python in CPU env)
#   CUDA_VENV, CUDA_PYTHON (bin/python in CUDA env)

set -e
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="$(dirname "$SETUP_DIR")"
REQ_FILE="$ARTIFACT_ROOT/requirements.txt"

CPU_VENV="$ARTIFACT_ROOT/my_venv"
CUDA_VENV="$ARTIFACT_ROOT/my_venv_cuda"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"
TORCH_CPU_INDEX="${TORCH_CPU_INDEX:-https://download.pytorch.org/whl/cpu}"

# Prefer the system python (/usr/bin/python3) over whatever `python3` resolves
# to on $PATH. Anaconda installs a python3 in $PATH whose binary lives outside
# the venv (symlink target ~/.anaconda3/...), and SGX cannot follow symlinks
# out of the mounted venv_dir at runtime.
if [[ -z "${PYTHON_BIN_HOST:-}" ]]; then
  if [[ -x /usr/bin/python3 ]]; then
    PYTHON_BIN_HOST=/usr/bin/python3
  else
    PYTHON_BIN_HOST=python3
  fi
fi

_setup_venv() {
  local venv_path="$1" wheel_index="$2" label="$3"
  if [[ -x "$venv_path/bin/python" ]]; then
    return 0
  fi
  echo "[setup] Creating $label venv at $venv_path ..."
  "$PYTHON_BIN_HOST" -m venv "$venv_path"
  "$venv_path/bin/pip" install --upgrade pip
  echo "[setup] Installing torch from $wheel_index ..."
  "$venv_path/bin/pip" install torch torchvision --index-url "$wheel_index"
  echo "[setup] Installing $REQ_FILE ..."
  "$venv_path/bin/pip" install -r "$REQ_FILE"
  # CUDA-only extra: cupy is needed by shmio.py to pin SHM for GPU DMA on the
  # backbone side. Skipped for the CPU venv since tee_server.py never pins.
  if [[ "$label" == "CUDA" ]]; then
    echo "[setup] Installing cupy-cuda12x (GPU side only) ..."
    "$venv_path/bin/pip" install cupy-cuda12x
  fi
  echo "[setup] $label venv ready."
}

_setup_venv "$CPU_VENV"  "$TORCH_CPU_INDEX"  "CPU"
_setup_venv "$CUDA_VENV" "$TORCH_CUDA_INDEX" "CUDA"

export CPU_VENV CUDA_VENV
export CPU_PYTHON="$CPU_VENV/bin/python"
export CUDA_PYTHON="$CUDA_VENV/bin/python"
echo "[setup] CPU_PYTHON  = $CPU_PYTHON"
echo "[setup] CUDA_PYTHON = $CUDA_PYTHON"
