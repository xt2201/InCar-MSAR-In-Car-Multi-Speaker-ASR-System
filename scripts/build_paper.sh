#!/usr/bin/env bash
# Build paper/main.pdf from paper/main.tex (pdflatex + bibtex)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/paper"

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "[ERROR] pdflatex not found. Install TeX Live or MacTeX, then re-run."
  exit 1
fi

name="main"
echo "[INFO] First pdflatex pass..."
pdflatex -interaction=nonstopmode "${name}.tex" >/dev/null
echo "[INFO] bibtex..."
bibtex "${name}" >/dev/null || true
echo "[INFO] Second + third pdflatex..."
pdflatex -interaction=nonstopmode "${name}.tex" >/dev/null
pdflatex -interaction=nonstopmode "${name}.tex" >/dev/null

if [[ -f "${name}.pdf" ]]; then
  echo "[OK] Output: $ROOT/paper/${name}.pdf"
  ls -la "${name}.pdf"
else
  echo "[ERROR] ${name}.pdf not produced. Check ${name}.log"
  exit 1
fi
