#!/bin/bash
# Artifact 2: End-to-end TEE-GPU hybrid inference (LST).
# For each of the three (model, dataset) configs reported in the paper, launches
# a fresh tee_server.py + backbone.py pair, waits for backbone.py to send its
# '__done__' sentinel, then tears tee_server.py down before the next config.
# (A single tee_server.py process processing multiple model swaps trips
# Gramine's malicious-host detection in TEE mode, so we keep one cfg per pair.)
#
# Modes:
#   bash run.sh                  # gpu-tee (DEFAULT — runs tee_server.py inside Gramine SGX)
#   bash run.sh gpu-cpu          # plain Python tee_server.py (use this if SGX is unavailable)
#   bash run.sh gpu-tee --build  # rebuild shm_bridge.so + SGX manifest first
#
# Output: ../results/<cfg>/{backbone.log, tee_server.log}
#         e.g. ../results/lst.8.vgg.gtsrb/backbone.log

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

MODE="${1:-gpu-tee}"
FORCE_BUILD=0
[[ "${2:-}" == "--build" ]] && FORCE_BUILD=1

if [[ "$MODE" != "gpu-cpu" && "$MODE" != "gpu-tee" ]]; then
  echo "Usage: bash run.sh [gpu-tee|gpu-cpu] [--build]"
  echo "  gpu-tee : tee_server.py inside Gramine SGX (default; requires SGX2)"
  echo "  gpu-cpu : tee_server.py as plain Python (use if SGX is unavailable)"
  exit 1
fi

LOG_DIR="$ROOT_DIR/results"
mkdir -p "$LOG_DIR"

echo "[*] Mode    : $MODE"
echo "[*] Log dir : $LOG_DIR"
if [[ "$MODE" == "gpu-tee" ]]; then
  echo "[*] Note    : if SGX is not available on this host, rerun with: bash run.sh gpu-cpu"
fi

# Bootstrap virtualenvs (idempotent — skipped if already created).
# Exports CPU_PYTHON, CUDA_PYTHON, CPU_VENV, CUDA_VENV.
source "$ROOT_DIR/shared/setup_envs.sh"

# Pre-download ViT-Base weights (required for gpu-tee — SGX has no network).
# Idempotent; skipped if shared/models/lst_vit/vit_base_patch16_224.pth exists.
"$CPU_PYTHON" -u "$ROOT_DIR/shared/prepare_vit_weights.py"

# Build shm_bridge.so (and Gramine SGX manifest for gpu-tee) when needed.
if [[ $FORCE_BUILD -eq 1 || ! -f "$SCRIPT_DIR/shm_bridge.so" ]]; then
  echo "[build] Building shm_bridge.so ..."
  make -C "$SCRIPT_DIR"
fi
if [[ "$MODE" == "gpu-tee" ]]; then
  # Detect a stale manifest whose baked-in entrypoint no longer matches the
  # bundle's CPU venv (happens if the manifest was first built against a
  # different venv on this host). If so, force a rebuild.
  STALE_MANIFEST=0
  if [[ -f "$SCRIPT_DIR/pytorch.manifest" ]]; then
    BAKED_EP="$(awk -F'"' '/^entrypoint *=/{print $2; exit}' "$SCRIPT_DIR/pytorch.manifest" || true)"
    if [[ -n "$BAKED_EP" && "$BAKED_EP" != "$CPU_PYTHON" ]]; then
      echo "[build] Stale manifest entrypoint ($BAKED_EP) != $CPU_PYTHON; rebuilding."
      STALE_MANIFEST=1
      rm -f "$SCRIPT_DIR/pytorch.manifest" "$SCRIPT_DIR/pytorch.manifest.sgx" "$SCRIPT_DIR/pytorch.sig"
    fi
  fi
  if [[ $FORCE_BUILD -eq 1 || $STALE_MANIFEST -eq 1 || ! -f "$SCRIPT_DIR/pytorch.manifest.sgx" ]]; then
    echo "[build] Building Gramine SGX manifest ..."
    make -C "$SCRIPT_DIR" sgx \
      VENV_DIR="$CPU_VENV" \
      PYTHON_BIN="$CPU_PYTHON"
  fi
fi

# Per-config launch: each config gets a fresh tee_server.py + backbone.py pair.
# Running multiple models inside a single tee_server.py process trips Gramine's
# malicious-host detection on model swap, so we spin them up and tear them down
# per-config.
CONFIGS=(
  "lst.8.vgg.gtsrb"
  "lst.8.resnet.cifar10"
  "lst.8.vit-base.cifar100"
)

# Wait up to TIMEOUT seconds for tee_server to exit cleanly (backbone sent
# __done__). If still alive, force-kill the gramine subtree so the next
# iteration can start with a fresh /dev/shm.
wait_or_kill_tee() {
  local pid=$1 timeout=30
  for _ in $(seq $timeout); do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      return 0
    fi
    sleep 1
  done
  echo "[*] tee_server (PID $pid) didn't exit in ${timeout}s; killing"
  pkill -KILL -P "$pid" 2>/dev/null || true
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

run_one_cfg() {
  local cfg="$1"
  local cfg_dir="$LOG_DIR/$cfg"
  mkdir -p "$cfg_dir"
  echo ""
  echo "================================================================="
  echo "[*] Config  : $cfg    (mode=$MODE)"
  echo "[*] Log dir : $cfg_dir"
  echo "================================================================="

  rm -f /dev/shm/* 2>/dev/null || true

  if [[ "$MODE" == "gpu-cpu" ]]; then
    ( cd "$SCRIPT_DIR" && nohup "$CPU_PYTHON" -u tee_server.py \
        > "$cfg_dir/tee_server.log" 2>&1 ) &
  else
    # Enclave cwd is "/", so pass the absolute path to tee_server.py.
    ( cd "$SCRIPT_DIR" && nohup gramine-sgx ./pytorch "$SCRIPT_DIR/tee_server.py" \
        > "$cfg_dir/tee_server.log" 2>&1 ) &
  fi
  local sd_pid=$!
  echo "[*] tee_server.py started (PID $sd_pid)"

  ( cd "$SCRIPT_DIR" && nohup "$CUDA_PYTHON" -u backbone.py --cfg "$cfg" \
      > "$cfg_dir/backbone.log" 2>&1 ) &
  local bb_pid=$!
  echo "[*] backbone.py started  (PID $bb_pid, --cfg $cfg)"

  wait "$bb_pid"
  local bb_rc=$?
  echo "[*] backbone.py exited (rc=$bb_rc); waiting for tee_server.py to finish ..."
  wait_or_kill_tee "$sd_pid"
  echo "[*] $cfg done"
}

for cfg in "${CONFIGS[@]}"; do
  run_one_cfg "$cfg"
done

echo ""
echo "[*] All configs complete. Per-config latency results:"
for cfg in "${CONFIGS[@]}"; do
  line="$(grep -E "^${cfg//./\\.}" "$LOG_DIR/$cfg/backbone.log" | tail -1 || true)"
  echo "    $cfg : ${line:-<no result>}"
done
echo ""
echo "[*] Logs: $LOG_DIR/<cfg>/{backbone.log, tee_server.log}"
