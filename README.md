# InCar-MSAR: In-Car Multi-Speaker ASR System

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Dataset: AISHELL-5](https://img.shields.io/badge/dataset-AISHELL--5-orange.svg)](https://www.openslr.org/159/)
[![CUDA 11.8](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-11-8-0-download-archive)

> *"Xe của bạn hiểu khi nhiều người nói cùng lúc."*

End-to-end pipeline for **in-car multi-speaker automatic speech recognition** using the AISHELL-5 dataset. Separates, transcribes, identifies, and understands commands from multiple simultaneous speakers in a vehicle cabin.

---

## Features

- **Speech Separation**: SepFormer (SpeechBrain) — separates 2–4 overlapping voices
- **ASR**: OpenAI Whisper — Mandarin Chinese transcription
- **Speaker Role Classification**: Driver / Passenger identification
- **Intent Engine**: 12 in-car command categories (climate, media, navigation, ...)
- **Streamlit Demo**: Real-time visualization with audio player and transcript table
- **Reproducible Evaluation**: WER, cpWER, Speaker Accuracy, Latency metrics

---

## Quick Start

```bash
# 1. Clone repository
git clone git@github.com:xt2201/InCar-MSAR-In-Car-Multi-Speaker-ASR-System.git
cd InCar-MSAR-In-Car-Multi-Speaker-ASR-System

# 2. Create Python environment
conda create -n incar-asr python=3.10
conda activate incar-asr

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set HuggingFace token (optional — speeds up data download)
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_...

# 5. Download AISHELL-5 data (auto: tries HuggingFace first, then OpenSLR)
#    dev + eval1 + eval2  (~7 GB total)
bash scripts/download_data.sh --all

# 6. Launch demo
streamlit run app.py
# Open http://localhost:8501
```

> **No dataset?** The script auto-downloads from the HuggingFace mirror
> [`xt2201/InCar-MSAR`](https://huggingface.co/datasets/xt2201/InCar-MSAR)
> when data is missing. Set `HF_TOKEN` in `.env` for authenticated access.

### Docker (Recommended)

```bash
# Build image
docker build -t incar-asr:v1.0 .

# Run demo
docker run --gpus all -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  incar-asr:v1.0

# Run evaluation
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  incar-asr:v1.0 bash scripts/run_eval.sh
```

---

## Installation

### Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ (recommended; CPU fallback available)
- ~8GB VRAM for Whisper-small + SepFormer

### Environment setup

```bash
# Conda
conda env create -f environment.yml
conda activate incar-asr

# Or pip
pip install -r requirements.txt
```

---

## Dataset: AISHELL-5

AISHELL-5 is mirrored on HuggingFace at
[`xt2201/InCar-MSAR`](https://huggingface.co/datasets/xt2201/InCar-MSAR)
(pre-materialized, fast) and on OpenSLR ([openslr.org/159](https://www.openslr.org/159/)) as the canonical source.

The download script tries HuggingFace first (requires `HF_TOKEN` in `.env`), then falls back to OpenSLR automatically.

```bash
# Dev + eval1 + eval2  (default for evaluation, ~7 GB total)
bash scripts/download_data.sh --all

# Individual splits
bash scripts/download_data.sh --eval1
bash scripts/download_data.sh --eval2
bash scripts/download_data.sh --noise   # ~12 GB, for augmentation

# Force one source
bash scripts/download_data.sh --all --hf-only
bash scripts/download_data.sh --all --openslr-only

# Environmental noise (large; mirrored on HuggingFace like other splits)
bash scripts/download_data.sh --noise
# (Maintainers) upload local data/noise to the HF dataset after OpenSLR download
# First run auto-creates .venv/ and installs huggingface-hub if missing.
bash scripts/upload_noise_to_hf.sh
```

**License**: CC BY-SA 4.0 — Attribution required. See [NOTICES.md](NOTICES.md).

Expected data structure after download:
```
data/
├── dev/              # validation split
│   ├── wav/          # 18 × 4-channel .wav (materialized)
│   └── text/         # 18 × .txt transcripts (SPK1: …, SPK2: …)
├── eval1/            # primary test benchmark
│   ├── wav/
│   └── text/
├── eval2/            # secondary test benchmark
│   ├── wav/
│   └── text/
└── noise/            # ~40h environmental noise (no transcripts)
    ├── 001/
    ├── 002/
    └── ...
```

---

## Usage

### Run Full Evaluation

Requires real AISHELL-5 under `data/eval1/` and `data/eval2/` (download script runs automatically if missing).

```bash
bash scripts/run_eval.sh              # full run: eval1 + eval2
bash scripts/run_eval.sh --quick      # n=5 smoke test
bash scripts/run_eval.sh --eval1      # eval1 only
bash scripts/run_eval.sh --eval2      # eval2 only
bash scripts/run_eval.sh --cpu        # force CPU mode
```

Results saved to `outputs/metrics/`. Generate PDF + tables: `bash scripts/build_paper.sh` after `bash scripts/generate_tables.py` (or use `run_eval.sh`, which runs table generation at the end).

### Evaluate Specific Split

```bash
# Single-channel baseline
python evaluate.py --split eval1 --mode baseline --n 100

# Full pipeline
python evaluate.py --split eval1 --mode pipeline --n 100

# Eval2 benchmark
python evaluate.py --split eval2 --mode pipeline
```

### Compute cpWER

```bash
python scripts/compute_cpwer.py \
  --hyp outputs/metrics/wer_pipeline_eval1.csv \
  --ref-dir data/eval1/text/
```

### Latency Benchmark

```bash
# Compare model sizes
python scripts/benchmark_latency.py --compare-models --n-chunks 50

# Single model
python scripts/benchmark_latency.py --model-size small --n-chunks 50
```

### Ablation Studies

```bash
# All ablations
python scripts/run_ablation.py --all --n 50

# Specific ablation
python scripts/run_ablation.py --study no-separation --n 50
```

---

## Configuration

Edit `configs/default.yaml` to change:

| Parameter | Default | Description |
|---|---|---|
| `audio.chunk_size_sec` | 2.0 | Streaming chunk duration |
| `asr.model` | `openai/whisper-small` | Whisper model size |
| `asr.beam_size` | 2 | ASR beam width |
| `separation.n_speakers` | 2 | Expected speakers |
| `speaker.method` | `hybrid` | `rule`, `ecapa`, or `hybrid` |

---

## Results

### CPU Benchmark (Measured, No GPU Required)

Measured on Apple M-series CPU with **Whisper-tiny** + **ChannelSelect** separation:

| System | Latency mean (ms/2s-chunk) | Latency P95 | RTF |
|---|---|---|---|
| ChannelSelect + Whisper-tiny | **218ms** | 220ms | 0.11x |
| Full pipeline (sep+ASR+role+intent) | ~462ms | ~520ms | 0.23x |

> RTF < 1.0 = faster than real-time. Both modes run in real-time on CPU.

### WER on AISHELL-5 (Real Data Required)

To get meaningful WER/cpWER numbers you need the real AISHELL-5 dataset:

```bash
bash scripts/download_data.sh --dev --eval1
python evaluate.py --split eval1 --mode baseline --n 100
python evaluate.py --split eval1 --mode pipeline --n 100
```

Expected results on real AISHELL-5 data (based on proposal targets):

| System | WER (%) | cpWER (%) | Speaker Acc. (%) |
|---|---|---|---|
| Single-ch ASR (Baseline) | ~45 | ~48 | ~62 |
| ChannelSelect + Whisper-tiny (CPU) | TBD | TBD | TBD |
| SepFormer + Whisper-small (GPU) | <30 (target) | <33 (target) | >80 (target) |

> Run `bash scripts/download_data.sh --all` then `bash scripts/run_eval.sh` for real benchmark numbers. The script auto-downloads from HuggingFace/OpenSLR if data is missing.

---

## Project Structure

```
thesis/
├── src/
│   ├── asr/           # Whisper ASR module
│   ├── separation/    # SepFormer / ConvTasNet
│   ├── speaker/       # ECAPA-TDNN + rule-based classifier
│   ├── intent/        # Rule-based intent engine
│   ├── pipeline/      # Main orchestrator + data loader
│   ├── evaluation/    # Metrics (WER, cpWER, SI-SNRi)
│   └── utils/         # Audio I/O, config, logging
├── configs/           # YAML configuration files
├── notebooks/         # EDA and analysis notebooks
├── tests/             # Unit tests (pytest)
├── scripts/           # Shell scripts
├── paper/             # LaTeX paper + references
├── outputs/           # Experiment results (gitignored)
│   ├── metrics/
│   ├── figures/
│   └── tables/
├── data/              # AISHELL-5 data (gitignored)
├── app.py             # Streamlit demo
├── evaluate.py        # Full evaluation entry-point
├── scripts/
│   ├── download_data.sh          # Data download (HF + OpenSLR)
│   ├── run_eval.sh               # Full benchmark pipeline
│   ├── build_paper.sh            # LaTeX → PDF
│   ├── generate_tables.py        # LaTeX result tables
│   ├── materialize_aishell5_flat.py  # Convert OpenSLR tree → flat wav/text
│   ├── benchmark_latency.py      # Per-chunk latency benchmark
│   ├── compute_cpwer.py          # Standalone cpWER tool
│   ├── run_ablation.py           # Ablation study runner
│   └── evaluate_separation.py    # Separation quality metrics
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Citation

```bibtex
@inproceedings{incar_msar_2026,
  title   = {{InCar-MSAR}: An End-to-End Pipeline for In-Car Multi-Speaker
             Automatic Speech Recognition with {AISHELL-5}},
  author  = {[Author]},
  year    = {2026},
  note    = {Available at: \url{https://github.com/[repo]}}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

### Third-Party Licenses

| Component | License | Notes |
|---|---|---|
| AISHELL-5 | CC BY-SA 4.0 | Attribution required |
| OpenAI Whisper | MIT | |
| SpeechBrain | Apache 2.0 | |
| Asteroid | MIT | |

See [NOTICES.md](NOTICES.md) for full third-party attribution.
