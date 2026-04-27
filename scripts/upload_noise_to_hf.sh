#!/usr/bin/env bash
# ============================================================
# Upload local data/noise/ to HuggingFace dataset xt2201/InCar-MSAR
# (path_in_repo: noise/). Large (~10-20 GB) - can take a long time.
#
# Uses HfApi.upload_large_folder (not upload_folder): multiple commits, local
# cache under data/.cache/huggingface/, resumable after interrupt — avoids
# single huge commits that often stall mid-upload on slow or flaky links.
#
# Optional env:
#   HF_UPLOAD_WORKERS       parallel workers (default: 2; lower if unstable)
#   HF_UPLOAD_REPORT_EVERY  status print interval in seconds (default: 30)
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

# Resolve Python: $PYTHON, then project .venv, then venv, then system python3
: "${PYTHON:=}"
if [[ -z "${PYTHON}" ]]; then
  if [[ -x "${PROJECT_ROOT}/.venv/bin/python3" ]]; then
    PYTHON="${PROJECT_ROOT}/.venv/bin/python3"
  elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${PROJECT_ROOT}/.venv/bin/python"
  elif [[ -x "${PROJECT_ROOT}/venv/bin/python3" ]]; then
    PYTHON="${PROJECT_ROOT}/venv/bin/python3"
  else
    PYTHON="python3"
  fi
fi
if [[ -x "${PYTHON}" ]]; then
  :
elif command -v "${PYTHON}" &>/dev/null; then
  PYTHON="$(command -v "${PYTHON}")"
else
  log_error "Python not found: ${PYTHON}. Set PYTHON=/path/to/python3 or create .venv in project root."
  exit 1
fi

ensure_huggingface_hub() {
  if "${PYTHON}" -c "import huggingface_hub" 2>/dev/null; then
    return 0
  fi
  log_warn "huggingface_hub not in this environment; using project .venv (one-time setup)."
  local vpy="${PROJECT_ROOT}/.venv/bin/python3"
  if [[ ! -x "${vpy}" ]]; then
    log_info "Creating ${PROJECT_ROOT}/.venv..."
    python3 -m venv "${PROJECT_ROOT}/.venv"
  fi
  "${PROJECT_ROOT}/.venv/bin/pip" install -q "pip>=24" "huggingface-hub>=0.25.0"
  PYTHON="${vpy}"
  if ! "${PYTHON}" -c "import huggingface_hub" 2>/dev/null; then
    log_error "Could not import huggingface_hub after .venv install. Run: ${PROJECT_ROOT}/.venv/bin/pip install huggingface-hub"
    exit 1
  fi
}
ensure_huggingface_hub
export HF_TOKEN
export NOISE_DIR
export PROJECT_ROOT
"${PYTHON}" - <<'PY'
import os
import time
from pathlib import Path

from huggingface_hub import HfApi

repo_id = "xt2201/InCar-MSAR"
data_root = Path(os.environ["PROJECT_ROOT"]) / "data"
noise_dir = Path(os.environ["NOISE_DIR"]).resolve()
if not data_root.is_dir():
    raise SystemExit(f"Missing data root: {data_root}")
# upload_large_folder has no path_in_repo; paths in repo = relative to data_root
# → files under data/noise/... become noise/... on the Hub (matches snapshot_download).
if not str(noise_dir).startswith(str(data_root.resolve())):
    raise SystemExit(f"noise dir must live under {data_root}, got {noise_dir}")

api = HfApi(token=os.environ["HF_TOKEN"])
if not hasattr(api, "upload_large_folder"):
    raise SystemExit("huggingface_hub too old: pip install 'huggingface-hub>=0.25' in this environment.")

workers = int(os.environ.get("HF_UPLOAD_WORKERS", "2"))
report_every = int(os.environ.get("HF_UPLOAD_REPORT_EVERY", "30"))

t0 = time.time()
api.upload_large_folder(
    repo_id=repo_id,
    folder_path=str(data_root),
    repo_type="dataset",
    allow_patterns=["noise/**"],
    ignore_patterns=[".*", "**/.DS_Store"],
    num_workers=workers,
    print_report=True,
    print_report_every=max(5, report_every),
)
print(f"Done in {time.time() - t0:.0f}s — see https://huggingface.co/datasets/{repo_id} (tree: noise/)")
PY

log_info "Complete. Others can run: bash scripts/download_data.sh --noise --hf-only"
