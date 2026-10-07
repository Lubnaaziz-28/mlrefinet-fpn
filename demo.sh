#!/usr/bin/env bash
# demo.sh — Auto-detect CUDA and launch the MLRefineFPN live detection UI.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---------- CUDA auto-detection ----------
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[demo] NVIDIA GPU detected — launching with CUDA."
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
else
    echo "[demo] No NVIDIA GPU detected — falling back to CPU."
fi

# ---------- Checkpoint ----------
CHECKPOINT="${CHECKPOINT:-weights/mlrefinet.pth}"
if [ ! -f "$CHECKPOINT" ]; then
    echo "[demo] Checkpoint '$CHECKPOINT' not found — running with untrained model for a live demo."
fi

# ---------- Install deps if missing ----------
if ! python -c "import gradio" >/dev/null 2>&1; then
    echo "[demo] Installing Python dependencies..."
    pip install --quiet -r requirements.txt
fi

# ---------- Launch ----------
echo "[demo] Starting Gradio live detection on http://localhost:7860"
exec python app.py --config configs/voc.yaml --checkpoint "$CHECKPOINT" --server-port 7860 --server-name 0.0.0.0