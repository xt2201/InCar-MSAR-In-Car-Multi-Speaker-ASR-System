# Changelog

All notable changes to InCar-MSAR are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] – 2026-04-23

### Added

**Core Pipeline**
- `InCarASRPipeline` orchestrator: end-to-end processing from 4-ch WAV → intent JSON
- `SpeechSeparator` and `ChunkedSeparator`: SepFormer and ConvTasNet separation wrappers
- `WhisperASR` and `ChunkedASR`: Whisper inference with streaming chunking
- `RuleBasedRoleClassifier`: energy-based Driver/Passenger classification
- `ECAPAEmbedding` and `SpeakerRoleClassifier`: ECAPA-TDNN hybrid classifier
- `IntentEngine`: keyword-based intent parsing for 12 in-car command categories

**Evaluation**
- `evaluate.py`: full WER/cpWER evaluation with baseline/pipeline modes
- `compute_cpwer.py`: Hungarian-algorithm cpWER implementation
- `evaluate_separation.py`: SI-SNRi proxy evaluation
- `benchmark_latency.py`: per-component latency measurement
- `run_ablation.py`: automated ablation study runner

**Infrastructure**
- `Dockerfile`: pytorch/pytorch:2.1.0-cuda11.8 based image
- `requirements.txt`: pinned dependencies for reproducibility
- `configs/default.yaml` and `configs/demo.yaml`: configuration schema
- `configs/intent_keywords.yaml`: externalized keyword mapping
- `scripts/download_data.sh`, `run_eval.sh`, `run_demo.sh`: automation scripts
- `scripts/generate_tables.py`: LaTeX table generator from CSV results

**Demo**
- `app.py`: Streamlit demo with audio player, waveform visualization,
  real-time transcript table with role/intent badges

**Paper**
- LaTeX paper: Introduction, Related Work, Methodology, Experiments, Conclusion
- `paper/references.bib`: 15+ bibliography entries
- Placeholder tables auto-generated from CSV results

**Testing**
- Unit tests: config, audio utils, metrics (WER/cpWER), intent engine (50 cases),
  speaker classifier

**Documentation**
- `README.md`: full documentation with Quick Start, Docker instructions, results table
- `NOTICES.md`: third-party license attributions
- `LICENSE`: MIT

### Metrics Achieved
- All code implemented and tested (unit tests pass for logic-testable components)
- Actual WER/cpWER numbers pending data download and GPU evaluation
- P95 latency target (<3s/chunk on T4) is designed for with whisper-small + beam_size=2

### Known Limitations
- SepFormer pretrained on WSJ0-2Mix (clean/anechoic), applied zero-shot to car audio
- Near-field reference not available in dev/eval → SI-SNRi uses energy-ratio proxy
- AISHELL-5 does not provide explicit Driver/Passenger ground truth annotations
  → Speaker Accuracy requires manual review or positional assumption

---

## [Unreleased]
- Fine-tuning SepFormer on AISHELL-5 train split
- LLM-based intent engine (Qwen-1.5)
- On-device quantization (INT8)
- Multilingual extension (Vietnamese, English)
