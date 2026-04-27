#!/usr/bin/env bash
# ============================================================
# Launch Streamlit demo application
# Usage: bash scripts/run_demo.sh [--port 8501] [--cpu]
# ============================================================
set -eu

PORT=8501
USE_CPU=false

for arg in "$@"; do
    case $arg in
        --port=*) PORT="${arg#*=}" ;;
        --cpu)    USE_CPU=true ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

if [ "${USE_CPU}" == "true" ]; then
    export CUDA_VISIBLE_DEVICES=""
    echo "[INFO] Running in CPU mode (CUDA disabled)"
fi

echo "============================================"
echo "In-Car Multi-Speaker ASR – Demo"
echo "URL: http://localhost:${PORT}"
echo "============================================"

# Check if data is available
if [ ! -d "data/dev/wav" ]; then
    echo "[WARN] No data found. Running download..."
    bash scripts/download_data.sh --dev
fi

# Launch Streamlit
streamlit run app.py \
    --server.port "${PORT}" \
    --server.address "0.0.0.0" \
    --server.headless true \
    --browser.gatherUsageStats false
