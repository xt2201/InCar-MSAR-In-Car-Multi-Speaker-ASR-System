#!/usr/bin/env bash
# ============================================================
# Upload local data/noise/ to HuggingFace dataset xt2201/InCar-MSAR
# (path_in_repo: noise/). Large (~10-20 GB) - can take a long time.
#
# Prereq: .env with HF_TOKEN; run after: bash scripts/download_data.sh --noise
# Usage: bash scripts/upload_noise_to_hf.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "./.env"
  set +a
fi

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

NOISE_DIR="${PROJECT_ROOT}/data/noise"
[[ -d "${NOISE_DIR}" ]] || { log_error "Missing ${NOISE_DIR}"; exit 1; }
n_wav=$(find "${NOISE_DIR}" -name "*.wav" 2>/dev/null | wc -l | tr -d ' ')
if [[ "${n_wav}" -eq 0 ]]; then
  log_error "No .wav under ${NOISE_DIR}. Download noise first: bash scripts/download_data.sh --noise"
  exit 1
fi
log_info "Found ${n_wav} WAV files under data/noise/"

if [[ -z "${HF_TOKEN:-}" ]]; then
  log_error "Set HF_TOKEN in .env (or export HF_TOKEN=...)"
  exit 1
fi

log_info "Uploading to HuggingFace xt2201/InCar-MSAR (path: noise/) - this may take a long time..."
: "${PYTHON:=python3}"
if ! command -v "${PYTHON}" &>/dev/null; then
  log_error "Python not found. Set PYTHON=/path/to/venv/bin/python3"
  exit 1
fi
if ! "${PYTHON}" -c "import huggingface_hub" 2>/dev/null; then
  log_error "Missing huggingface_hub. Install: pip install -r requirements.txt"
  exit 1
fi
export HF_TOKEN
export NOISE_DIR
"${PYTHON}" - <<'PY'
import os, time
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
t0 = time.time()
url = api.upload_folder(
    folder_path=os.environ["NOISE_DIR"],
    path_in_repo="noise",
    repo_id="xt2201/InCar-MSAR",
    repo_type="dataset",
    commit_message="Add environmental noise split (no transcripts)",
    token=os.environ["HF_TOKEN"],
    ignore_patterns=[".*", "**/.DS_Store"],
)
print(f"Done in {time.time() - t0:.0f}s: {url}")
PY

log_info "Complete. Others can run: bash scripts/download_data.sh --noise --hf-only"
