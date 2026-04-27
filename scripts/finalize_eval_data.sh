#!/usr/bin/env bash
# ============================================================
# Finalize eval1 and eval2 data after OpenSLR tarballs finish downloading.
# Run this after Eval1_dl.tar.gz and Eval2_dl.tar.gz are complete.
#
# Usage:
#   bash scripts/finalize_eval_data.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

[[ -f ".env" ]] && export "$(grep '^HF_TOKEN=' .env | head -1)" 2>/dev/null || true

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

finalize_split() {
  local split="$1"
  local tarball="data/${split^}_dl.tar.gz"  # e.g. Eval1_dl.tar.gz

  # Normalize: eval1 -> Eval1_dl.tar.gz
  case "$split" in
    eval1) tarball="data/Eval1_dl.tar.gz" ;;
    eval2) tarball="data/Eval2_dl.tar.gz" ;;
    dev)   tarball="data/Dev_dl.tar.gz" ;;
    *) log_error "Unknown split: $split"; return 1 ;;
  esac

  local raw_dir="data/${split}_openslr_raw"
  local dest_dir="data/${split}"

  # Already done?
  local n
  n=$(find "${dest_dir}/wav" -name "*.wav" 2>/dev/null | wc -l | tr -d ' ') || n=0
  if [[ "${n}" -gt 0 ]]; then
    log_info "${split} already materialized (${n} wav files). Skipping."
    return 0
  fi

  # Wait for download to complete
  if [[ ! -f "${tarball}" ]]; then
    log_warn "Tarball not found: ${tarball}. Skip ${split}."
    return 1
  fi

  log_info "Validating ${tarball}..."
  if ! gzip -t "${tarball}" 2>/dev/null; then
    log_error "Tarball incomplete or corrupt: ${tarball}"
    log_error "Wait for download to finish, then re-run this script."
    return 1
  fi

  log_info "Extracting ${tarball} → ${raw_dir}/"
  mkdir -p "${raw_dir}"
  tar -xzf "${tarball}" -C "${raw_dir}" --strip-components=1 2>/dev/null || \
    tar -xzf "${tarball}" -C "${raw_dir}"
  rm -f "${tarball}"

  log_info "Materializing ${split}: ${raw_dir} → ${dest_dir}"
  python scripts/materialize_aishell5_flat.py \
    --source "${raw_dir}" \
    --dest   "${dest_dir}" \
    --label  "${split}"
  rm -rf "${raw_dir}"

  log_info "✓ ${split} ready: ${dest_dir}/wav/ + ${dest_dir}/text/"
}

# Upload to HuggingFace
upload_to_hf() {
  local split="$1"
  local dest_dir="data/${split}"
  [[ -z "${HF_TOKEN:-}" ]] && { log_warn "No HF_TOKEN; skip HF upload for ${split}."; return; }
  log_info "Uploading ${split} to HuggingFace xt2201/InCar-MSAR..."
  python - << PYEOF
from huggingface_hub import HfApi
import time
api = HfApi(token="${HF_TOKEN}")
t0 = time.time()
url = api.upload_folder(
    folder_path="${dest_dir}",
    path_in_repo="${split}",
    repo_id="xt2201/InCar-MSAR",
    repo_type="dataset",
    commit_message="Add ${split} materialized split",
    token="${HF_TOKEN}",
    ignore_patterns=[".*"],
)
print(f"[${split}] Done in {time.time()-t0:.0f}s")
PYEOF
}

for split in eval1 eval2; do
  finalize_split "${split}" && upload_to_hf "${split}" || true
done

log_info "Done."
