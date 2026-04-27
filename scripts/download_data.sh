#!/usr/bin/env bash
# ============================================================
# Download AISHELL-5 dataset (OpenSLR #159 + HuggingFace mirror).
# License: CC BY-SA 4.0 — https://www.openslr.org/159/
#
# Primary:   HuggingFace dataset repo  xt2201/InCar-MSAR
#            (pre-materialized 4-ch wav + text, fast, no extraction)
# Fallback:  OpenSLR mirrors (raw tarballs → materialize on-the-fly)
#
# IMPORTANT (macOS APFS):
#   The filesystem is case-insensitive — data/Eval1 == data/eval1.
#   Tarballs are extracted to data/<split>_openslr_raw/ (different name)
#   then materialized to data/<split>/, then raw dir is deleted.
#
# Usage:
#   bash scripts/download_data.sh              # dev + eval1 (default)
#   bash scripts/download_data.sh --all        # dev + eval1 + eval2
#   bash scripts/download_data.sh --all-except-train  # dev+eval1+eval2+noise
#   bash scripts/download_data.sh --full       # all including train (~51GB)
#   bash scripts/download_data.sh --hf-only    # HuggingFace only, skip OpenSLR
#   bash scripts/download_data.sh --openslr-only  # OpenSLR only, skip HF
#   bash scripts/download_data.sh --dev --eval1  # individual splits
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_blue()  { echo -e "${BLUE}[HF]${NC} $*"; }

# HuggingFace repo & token
HF_REPO="xt2201/InCar-MSAR"
HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" && -f "${PROJECT_ROOT}/.env" ]]; then
  HF_TOKEN="$(grep '^HF_TOKEN=' "${PROJECT_ROOT}/.env" 2>/dev/null | cut -d= -f2- | tr -d '"' || true)"
fi

# OpenSLR mirrors (tried in order)
MIRRORS=(
  "https://www.openslr.org/resources/159"
  "https://openslr.magicdatatech.com/resources/159"
  "https://us.openslr.org/resources/159"
)

# Map logical split name → tarball filename on server (capitalized on OpenSLR)
archive_name_for() {
  case "$1" in
    dev)   echo "Dev.tar.gz" ;;
    eval1) echo "Eval1.tar.gz" ;;
    eval2) echo "Eval2.tar.gz" ;;
    train) echo "train.tar.gz" ;;
    noise) echo "noise.tar.gz" ;;
    *)     echo "" ;;
  esac
}

# --------------------------------------------------------------------------
# Flags
# --------------------------------------------------------------------------
DOWNLOAD_DEV=false
DOWNLOAD_EVAL1=false
DOWNLOAD_EVAL2=false
DOWNLOAD_TRAIN=false
DOWNLOAD_NOISE=false
HF_ONLY=false
OPENSLR_ONLY=false

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]
  --dev               Download dev split (~1.8 GB)
  --eval1             Download eval1 split (~1.6 GB)
  --eval2             Download eval2 split (~1.8 GB)
  --train             Download train split (~51 GB, very large!)
  --noise             Download noise split (~12 GB)
  --all               dev + eval1 + eval2
  --all-except-train  dev + eval1 + eval2 + noise
  --full              dev + eval1 + eval2 + train + noise
  --hf-only           Use HuggingFace mirror only
  --openslr-only      Use OpenSLR mirrors only (skips HF)
  --help              Show this help
EOF
  exit 0
}

for arg in "$@"; do
  case $arg in
    --dev)               DOWNLOAD_DEV=true ;;
    --eval1)             DOWNLOAD_EVAL1=true ;;
    --eval2)             DOWNLOAD_EVAL2=true ;;
    --train)             DOWNLOAD_TRAIN=true ;;
    --noise)             DOWNLOAD_NOISE=true ;;
    --all)               DOWNLOAD_DEV=true; DOWNLOAD_EVAL1=true; DOWNLOAD_EVAL2=true ;;
    --all-except-train)  DOWNLOAD_DEV=true; DOWNLOAD_EVAL1=true; DOWNLOAD_EVAL2=true; DOWNLOAD_NOISE=true ;;
    --full)              DOWNLOAD_DEV=true; DOWNLOAD_EVAL1=true; DOWNLOAD_EVAL2=true; DOWNLOAD_TRAIN=true; DOWNLOAD_NOISE=true ;;
    --hf-only)           HF_ONLY=true ;;
    --openslr-only)      OPENSLR_ONLY=true ;;
    --help)              usage ;;
    *) log_error "Unknown option: $arg"; usage ;;
  esac
done

# Default: dev + eval1
if ! $DOWNLOAD_DEV && ! $DOWNLOAD_EVAL1 && ! $DOWNLOAD_EVAL2 && \
   ! $DOWNLOAD_TRAIN && ! $DOWNLOAD_NOISE; then
  DOWNLOAD_DEV=true
  DOWNLOAD_EVAL1=true
fi

mkdir -p "${DATA_DIR}"

# --------------------------------------------------------------------------
# HuggingFace download helper
# --------------------------------------------------------------------------
hf_download_split() {
  local split="$1"
  local dest_dir="${DATA_DIR}/${split}"

  if [[ -n "$HF_TOKEN" ]]; then
    log_blue "Trying HuggingFace: ${HF_REPO} → ${split}/"
    python - <<PYEOF 2>/dev/null && return 0 || true
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
import os, shutil
token = "${HF_TOKEN}"
repo = "${HF_REPO}"
dest = "${dest_dir}"
os.makedirs(dest, exist_ok=True)
try:
    local = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=dest,
        allow_patterns=["${split}/**"],
        token=token,
    )
    print(f"[HF] Downloaded {local}")
except Exception as e:
    print(f"[HF] Failed: {e}")
    raise
PYEOF
  fi
  return 1
}

# --------------------------------------------------------------------------
# OpenSLR download + materialize helper
# --------------------------------------------------------------------------
try_wget() {
  local url="$1" dest="$2"
  wget --no-check-certificate -c --timeout=60 --tries=3 \
    --progress=bar:force -O "${dest}" "${url}" 2>/dev/null
}

try_curl() {
  local url="$1" dest="$2"
  curl -k -L -C - --max-time 0 --retry 3 --retry-delay 5 \
    --progress-bar -o "${dest}" "${url}" 2>/dev/null
}

openslr_download_split() {
  local split="$1"
  local dest_dir="${DATA_DIR}/${split}"
  local filename
  filename="$(archive_name_for "${split}")"
  [[ -z "${filename}" ]] && { log_error "Unknown split: ${split}"; return 1; }

  # Extract to <split>_openslr_raw/ to avoid macOS case-insensitive collision
  # (data/Eval1 == data/eval1 on APFS; use data/eval1_openslr_raw as staging)
  local raw_dir="${DATA_DIR}/${split}_openslr_raw"
  local dest_tar="${DATA_DIR}/${filename}"

  log_info "Downloading from OpenSLR: ${filename}"
  local ok=false
  for mirror in "${MIRRORS[@]}"; do
    log_info "  Trying: ${mirror}/${filename}"
    if try_wget "${mirror}/${filename}" "${dest_tar}" || \
       try_curl "${mirror}/${filename}" "${dest_tar}"; then
      ok=true; break
    fi
  done

  if ! $ok; then
    log_error "All mirrors failed for ${filename}."
    rm -f "${dest_tar}"
    return 1
  fi

  if ! gzip -t "${dest_tar}" 2>/dev/null; then
    log_error "Downloaded archive is corrupted: ${dest_tar}. Remove and retry."
    rm -f "${dest_tar}"
    return 1
  fi

  log_info "Extracting ${filename} → ${raw_dir}/"
  mkdir -p "${raw_dir}"
  # Noise tarballs extract as noise/001/, etc.; eval tarballs as Eval1/001/ etc.
  # We strip the top-level directory into raw_dir
  tar -xzf "${dest_tar}" -C "${raw_dir}" --strip-components=1 2>/dev/null || \
    tar -xzf "${dest_tar}" -C "${raw_dir}" 2>/dev/null
  rm -f "${dest_tar}"

  # Materialize: stack DX01-04 → 4-ch wav + TextGrid → text/
  log_info "Materializing ${split}: ${raw_dir} → ${dest_dir}"
  python "${SCRIPT_DIR}/materialize_aishell5_flat.py" \
    --source "${raw_dir}" \
    --dest   "${dest_dir}" \
    --label  "${split}"

  # If noise (no TextGrid), just move wav structure as-is
  if [[ "${split}" == "noise" ]]; then
    # noise has no transcripts — skip materialize, just link as-is
    log_warn "Noise split: no transcripts. Raw session wav files kept at ${raw_dir}"
    rm -rf "${dest_dir}" 2>/dev/null || true
    mv "${raw_dir}" "${dest_dir}"
    return 0
  fi

  # Remove staging dir after successful materialize
  rm -rf "${raw_dir}"
  log_info "✓ ${split} ready: ${dest_dir}/wav + ${dest_dir}/text"
}

# --------------------------------------------------------------------------
# Main: download_split selects HF first, then OpenSLR fallback
# --------------------------------------------------------------------------
download_split() {
  local split="$1"
  local dest_dir="${DATA_DIR}/${split}"

  # Already have materialized wav/ data?
  local wav_count=0
  if [[ -d "${dest_dir}/wav" ]]; then
    wav_count=$(find "${dest_dir}/wav" -name "*.wav" 2>/dev/null | wc -l | tr -d ' ')
  elif [[ -d "${dest_dir}" ]]; then
    wav_count=$(find "${dest_dir}" -name "*.wav" 2>/dev/null | wc -l | tr -d ' ')
  fi
  if [[ "${wav_count}" -gt 0 ]]; then
    log_warn "${split} already has ${wav_count} .wav files — skip download."
    return 0
  fi

  # 1) Try HuggingFace (pre-materialized, fast)
  if ! $OPENSLR_ONLY; then
    if hf_download_split "${split}"; then
      log_info "✓ ${split} from HuggingFace"
      return 0
    fi
    log_warn "HuggingFace unavailable or no token; falling back to OpenSLR."
  fi

  # 2) Fallback: OpenSLR
  if ! $HF_ONLY; then
    openslr_download_split "${split}"
  else
    log_error "HF-only mode but HuggingFace failed for ${split}."
    return 1
  fi
}

verify_split() {
  local split="$1"
  local dir="${DATA_DIR}/${split}"
  local n=0
  [[ -d "${dir}/wav" ]] && n=$(find "${dir}/wav" -name "*.wav" 2>/dev/null | wc -l | tr -d ' ') || \
    n=$(find "${dir}" -name "*.wav" 2>/dev/null | wc -l | tr -d ' ')
  log_info "  ${split}: ${n} WAV files"
}

# --------------------------------------------------------------------------
# Download metadata from OpenSLR (small, optional)
# --------------------------------------------------------------------------
OPENSLR_META="${DATA_DIR}/openslr159"
mkdir -p "${OPENSLR_META}"
for meta in about.html info.txt; do
  [[ -s "${OPENSLR_META}/${meta}" ]] && continue
  for mirror in "${MIRRORS[@]}"; do
    try_wget "${mirror}/${meta}" "${OPENSLR_META}/${meta}" 2>/dev/null && break || true
  done
done

# --------------------------------------------------------------------------
# Execute
# --------------------------------------------------------------------------
log_info "AISHELL-5 Dataset — CC BY-SA 4.0 (https://www.openslr.org/159/)"
log_info "HuggingFace mirror: ${HF_REPO}"
log_info "Target: ${DATA_DIR}"
[[ -n "$HF_TOKEN" ]] && log_blue "HF_TOKEN detected." || log_warn "No HF_TOKEN — will use OpenSLR."
echo ""

$DOWNLOAD_DEV   && download_split "dev"
$DOWNLOAD_EVAL1 && download_split "eval1"
$DOWNLOAD_EVAL2 && download_split "eval2"
$DOWNLOAD_TRAIN && download_split "train"
$DOWNLOAD_NOISE && download_split "noise"

echo ""
log_info "=== Verification ==="
$DOWNLOAD_DEV   && verify_split "dev"   || true
$DOWNLOAD_EVAL1 && verify_split "eval1" || true
$DOWNLOAD_EVAL2 && verify_split "eval2" || true
$DOWNLOAD_NOISE && verify_split "noise" || true

echo ""
log_info "Setup complete. Next step: bash scripts/run_eval.sh"
