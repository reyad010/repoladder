#!/bin/bash
# Artifact 1: Genetic NAS for layer-reduction LST.
# For each (model, dataset) pair, launches tee_server.py + search.py, waits for
# the search to finish, then tears tee_server down before the next pair.
# Pairs run: vgg/gtsrb,  resnet/cifar10,  vit-base/cifar100.
#
# Modes:
#   bash run.sh                  # gpu-tee (DEFAULT — tee_server.py inside Gramine SGX)
#   bash run.sh gpu-cpu          # plain Python tee_server.py (use if SGX unavailable)
#   bash run.sh single           # single-process; side runs in-thread on CPU (no tee_server)
#
# Output: results/<model>_<dataset>/  (search.log, hall_of_fame.json, tee_server.log)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
TEE_DIR="$ROOT_DIR/artifact2_e2e"

MODE="${1:-gpu-tee}"

if [[ "$MODE" != "gpu-tee" && "$MODE" != "gpu-cpu" && "$MODE" != "single" ]]; then
  echo "Usage: bash run.sh [gpu-tee|gpu-cpu|single]"
  exit 1
fi
if [[ "$MODE" == "gpu-tee" ]]; then
  echo "[*] Note: if SGX is not available on this host, rerun with: bash run.sh gpu-cpu"
fi

# (model, dataset) pairs — matches backbone.py's default workload and Fig. 10.
PAIRS=(
  "vgg gtsrb"
  "resnet cifar10"
  "vit-base cifar100"
)
N_GEN=1    # debug mode (paper: 10)
NPOP=6     # debug mode (paper: 64)
BETA=30

# Bootstrap virtualenvs (idempotent — skipped if already created).
# Exports CPU_PYTHON, CUDA_PYTHON, CPU_VENV, CUDA_VENV.
source "$ROOT_DIR/shared/setup_envs.sh"

# Ensure .data exists before search.py runs — torchvision auto-downloads into
# this dir, but the dataloader's download path doesn't mkdir parents itself.
# (Reviewers may also `ln -sfn /path/to/datasets $SCRIPT_DIR/.data`.)
mkdir -p "$SCRIPT_DIR/.data"

# Pre-download ViT-Base weights (required for gpu-tee — SGX has no network).
# Idempotent; skipped if shared/models/lst_vit/vit_base_patch16_224.pth exists.
"$CPU_PYTHON" -u "$ROOT_DIR/shared/prepare_vit_weights.py"

# Build shm_bridge.so (and SGX manifest for gpu-tee) when needed.
if [[ "$MODE" != "single" ]]; then
  if [[ ! -f "$TEE_DIR/shm_bridge.so" ]]; then
    echo "[build] Building shm_bridge.so ..."
    make -C "$TEE_DIR"
  fi
  if [[ "$MODE" == "gpu-tee" ]]; then
    # Detect a stale manifest whose baked-in entrypoint no longer matches the
    # bundle's CPU venv (happens if the manifest was first built against a
    # different venv on this host). If so, force a rebuild.
    STALE_MANIFEST=0
    if [[ -f "$TEE_DIR/pytorch.manifest" ]]; then
      BAKED_EP="$(awk -F'"' '/^entrypoint *=/{print $2; exit}' "$TEE_DIR/pytorch.manifest" || true)"
      if [[ -n "$BAKED_EP" && "$BAKED_EP" != "$CPU_PYTHON" ]]; then
        echo "[build] Stale manifest entrypoint ($BAKED_EP) != $CPU_PYTHON; rebuilding."
        STALE_MANIFEST=1
        rm -f "$TEE_DIR/pytorch.manifest" "$TEE_DIR/pytorch.manifest.sgx" "$TEE_DIR/pytorch.sig"
      fi
    fi
    if [[ $STALE_MANIFEST -eq 1 || ! -f "$TEE_DIR/pytorch.manifest.sgx" ]]; then
      echo "[build] Building Gramine SGX manifest ..."
      make -C "$TEE_DIR" sgx \
        VENV_DIR="$CPU_VENV" \
        PYTHON_BIN="$CPU_PYTHON"
    fi
  fi
fi

run_one_pair() {
  local model="$1" dataset="$2"
  local log_dir="$SCRIPT_DIR/results/${model}_${dataset}"
  mkdir -p "$log_dir"
  echo ""
  echo "================================================================="
  echo "[*] Pair    : $model / $dataset    (mode=$MODE)"
  echo "[*] Log dir : $log_dir"
  echo "================================================================="

  if [[ "$MODE" == "single" ]]; then
    "$CUDA_PYTHON" -u "$SCRIPT_DIR/search.py" \
      --model_name "$model" --dataset "$dataset" \
      --n_generation "$N_GEN" --npop "$NPOP" --reward_beta "$BETA" \
      --test_device single \
      2>&1 | tee "$log_dir/search.log"
    return
  fi

  # Two-process mode: launch tee_server.py and search.py in parallel.
  # The shared-memory channel uses spinlocks, so whichever side starts first
  # will simply wait for the other; no explicit readiness handshake is needed.
  rm -f /dev/shm/* 2>/dev/null || true
  if [[ "$MODE" == "gpu-tee" ]]; then
    ( cd "$TEE_DIR" && nohup gramine-sgx ./pytorch "$TEE_DIR/tee_server.py" \
        > "$log_dir/tee_server.log" 2>&1 ) &
  else
    ( cd "$TEE_DIR" && nohup "$CPU_PYTHON" -u tee_server.py \
        > "$log_dir/tee_server.log" 2>&1 ) &
  fi
  local tee_pid=$!
  echo "[*] tee_server.py started (PID $tee_pid)"

  ( "$CUDA_PYTHON" -u "$SCRIPT_DIR/search.py" \
      --model_name "$model" --dataset "$dataset" \
      --n_generation "$N_GEN" --npop "$NPOP" --reward_beta "$BETA" \
      --test_device split \
      > "$log_dir/search.log" 2>&1 ) &
  local search_pid=$!
  echo "[*] search.py started      (PID $search_pid)"
  echo "[*] Both processes launched in parallel. Logs:"
  echo "    search     : $log_dir/search.log"
  echo "    tee_server : $log_dir/tee_server.log"

  # Wait for the search to complete (it sends '__done__' to tee_server on exit).
  wait "$search_pid"

  # search.py sends '__done__' when finished; reap tee_server.
  wait "$tee_pid" 2>/dev/null || true
  echo "[*] tee_server.py stopped (PID $tee_pid)"
}

for pair in "${PAIRS[@]}"; do
  run_one_pair $pair
done

echo ""
echo "[*] All pairs complete. Each results/<model>_<dataset>/ has the NAS hall_of_fame.json."
