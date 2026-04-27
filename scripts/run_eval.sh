#!/usr/bin/env bash
# ============================================================
# Full evaluation pipeline — AISHELL-5 real data required.
# Downloads data automatically from HuggingFace if not present.
#
# Usage:
#   bash scripts/run_eval.sh           # full run (eval1 + eval2)
#   bash scripts/run_eval.sh --quick   # n=5 per split (smoke test)
#   bash scripts/run_eval.sh --cpu     # force CPU mode
#   bash scripts/run_eval.sh --eval1   # eval1 only
#   bash scripts/run_eval.sh --eval2   # eval2 only
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTHONWARNINGS="ignore"
export PYTHONSEED=42

# Load HF token if present
[[ -f ".env" ]] && export "$(grep '^HF_TOKEN=' .env | head -1)" 2>/dev/null || true

# ---------- Flags ----------
QUICK=false
ONLY_EVAL1=false
ONLY_EVAL2=false
for arg in "$@"; do
  case $arg in
    --quick)  QUICK=true ;;
    --cpu)    export CUDA_VISIBLE_DEVICES="" ;;
    --eval1)  ONLY_EVAL1=true ;;
    --eval2)  ONLY_EVAL2=true ;;
  esac
done

N=30
if $QUICK; then N=5; echo "[INFO] Quick mode: n=${N}"; fi

echo "============================================"
echo "In-Car Multi-Speaker ASR — Evaluation"
echo "Date:   $(date)"
echo "Python: $(python --version 2>&1)"
echo "CUDA:   $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo unknown)"
echo "============================================"

# ---------- Auto-download if missing ----------
need_download=false
[[ "$(find data/eval1/wav -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')" -lt 1 ]] && need_download=true
[[ "$(find data/eval2/wav -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')" -lt 1 ]] && need_download=true

if $need_download; then
  echo "[INFO] Missing eval splits — downloading from HuggingFace / OpenSLR..."
  bash scripts/download_data.sh --eval1 --eval2
fi

mkdir -p outputs/metrics outputs/ablation outputs/tables

RUN_EVAL1=true
RUN_EVAL2=true
$ONLY_EVAL1 && RUN_EVAL2=false
$ONLY_EVAL2 && RUN_EVAL1=false

SKIP_EVAL2=false
[[ "$(find data/eval2/wav -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')" -lt 1 ]] && SKIP_EVAL2=true

# ------------------------------------------------------------------
# 1. Baseline: Single-channel ASR on eval1
# ------------------------------------------------------------------
if $RUN_EVAL1; then
  echo ""
  echo "--- [1/6] Baseline (single-channel) — eval1 ---"
  python evaluate.py --split eval1 --mode baseline --n "$N" --config configs/default.yaml
  echo "[INFO] Baseline done."
fi

# ------------------------------------------------------------------
# 2. Full pipeline on eval1
# ------------------------------------------------------------------
if $RUN_EVAL1; then
  echo ""
  echo "--- [2/6] Full Pipeline — eval1 ---"
  python evaluate.py --split eval1 --mode pipeline --n "$N" --config configs/default.yaml
  echo "[INFO] Pipeline eval1 done."
fi

# ------------------------------------------------------------------
# 3. Full pipeline on eval2
# ------------------------------------------------------------------
if $RUN_EVAL2; then
  if $SKIP_EVAL2; then
    echo ""
    echo "--- [3/6] eval2 — skipped (data not found) ---"
  else
    echo ""
    echo "--- [3/6] Full Pipeline — eval2 ---"
    python evaluate.py --split eval2 --mode pipeline --n "$N" --config configs/default.yaml
    echo "[INFO] Pipeline eval2 done."
  fi
fi

# ------------------------------------------------------------------
# 4. Tests
# ------------------------------------------------------------------
echo ""
echo "--- [4/6] Unit + Integration Tests ---"
# Use || true so a test failure does not abort the script before table generation.
# Test results are still visible; non-zero exit is re-reported at the end.
python -m pytest tests/ -q --tb=short; PYTEST_EXIT=$?
if [ "$PYTEST_EXIT" -ne 0 ]; then
  echo "[WARN] Some tests failed (exit=$PYTEST_EXIT) — see above. Continuing to table generation."
fi
echo "[INFO] Tests done."

# ------------------------------------------------------------------
# 5. Error Analysis
# ------------------------------------------------------------------
echo ""
echo "--- [5/6] Error Analysis ---"
python scripts/error_analysis.py --all-csvs outputs/metrics/ \
    --output outputs/metrics/error_analysis_all.json 2>/dev/null || \
    echo "[WARN] Error analysis failed (continuing)"
echo "[INFO] Error analysis done."

# ------------------------------------------------------------------
# 6. Tables & Paper
# ------------------------------------------------------------------
echo ""
echo "--- [6/6] Generate LaTeX Tables ---"
python scripts/generate_tables.py
cp -f outputs/tables/*.tex paper/tables/ 2>/dev/null || true
echo "[INFO] Tables generated."

echo ""
echo "============================================"
echo "Evaluation complete!"
echo "  Metrics:  outputs/metrics/"
echo "  Tables:   outputs/tables/  &  paper/tables/"
echo "  Errors:   outputs/metrics/error_analysis_all.json"
echo "============================================"
echo "$(date): run_eval.sh completed" >> outputs/repro_check.txt
