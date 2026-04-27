# Project Retrospective – InCar-MSAR

**Date**: 2026-04-27  
**Phase**: 2.0 (Stability, Ablation & Deployment Readiness)  
**Sprint Span**: S11–S14 (Phase 2 improvements)

---

## Phase 1 Summary (S1–S10)

Phase 1 delivered a working end-to-end pipeline with the following components: SepFormer-based speech separation, Whisper-tiny ASR, rule-based + ECAPA-TDNN speaker classification, YAML intent engine, and a Streamlit demo. Evaluation on AISHELL-5 eval1 (5 sessions, CPU) showed:

| System | WER (CER) | cpWER | Latency |
|---|---|---|---|
| Baseline (channel 0, no sep) | ~126% | ~126% | ~42s/session |
| Full pipeline (channel_select + whisper-tiny) | ~183% | ~183% | ~109s/session |

Both numbers exceeded 100% — a common outcome for Mandarin CER when Whisper hallucinates on long sessions or when the hypothesis is much longer than the reference.

---

## What Worked Well

### 1. Modular Architecture with Lazy Loading
The decision to organize the pipeline into discrete, independently-loadable modules
(`SpeechSeparator`, `WhisperASR`, `SpeakerRoleClassifier`, `IntentEngine`) proved
highly effective. Heavy models (Whisper, SepFormer, ECAPA) are loaded lazily,
which enables fast unit testing without GPU and allows the demo to initialize
incrementally.

### 2. Config-Driven Design
Externalizing all hyperparameters into `configs/default.yaml` made reproducibility
significantly easier. The OmegaConf library provided dot-access and structured validation.

### 3. cpWER with Hungarian Algorithm
Implementing cpWER correctly using `scipy.optimize.linear_sum_assignment` was a
key scientific contribution. The brute-force fallback for N≤4 provides an
independent check.

### 4. Intent Engine with External YAML
Externalizing keyword mappings to `configs/intent_keywords.yaml` enabled rapid
iteration without touching Python code.

### 5. Streamlit Demo Design
The split-panel UI (audio player on left, transcript table on right) proved
intuitive. Color-coding by speaker role (blue=Driver, orange=Passenger) and
intent icons (🌡️, ▶️, 🗺️) made the output readable without parsing JSON.

---

## Phase 2 Improvements

### 1. Whisper Hallucination Prevention (Critical Fix)

**Root cause**: Whisper-tiny operating on full sessions (often 3–10 minutes) generated
repetitive text on silence/noise segments. This inflated CER above 100% in both
baseline and pipeline evaluations.

**Fixes applied**:
- `no_speech_threshold=0.6`, `compression_ratio_threshold=2.4`, `condition_on_prev_tokens=False`
  set directly on `model.generation_config` (not `generate_kwargs`) to avoid
  Transformers 5.x crash.
- `WhisperASR` now auto-delegates audio longer than 30 seconds to `ChunkedASR` to prevent
  hallucination accumulation over silence.
- `_compression_ratio()` static method added to detect repetition loops post-generation.
- `max_length` cleared from `generation_config` to prevent the `max_new_tokens vs max_length` conflict.

### 2. Segment-Level Evaluation Mode

**Root cause**: Full-session evaluation fed minutes of audio to Whisper at once, producing
hallucinated long hypotheses that vastly inflated WER regardless of separation quality.

**Fix**: Added `--eval-mode segment` to `evaluate.py`. TextGrid annotations are parsed
by `materialize_aishell5_flat.py --export-segments` to produce per-utterance WAV clips
(stored in `data/{split}/segments/`). Each clip (typically 1–10s) is evaluated
independently, providing an accurate WER measurement isolated from hallucination bias.

### 3. ASR Model Upgrade

Default model upgraded from `openai/whisper-tiny` to `openai/whisper-small` with
`beam_size=2` in `configs/default.yaml`. Small has 244M parameters vs. 39M for tiny,
with significantly better Mandarin CER on far-field audio.

### 4. MVDR Beamforming Backend

Added `BeamformSeparator` in `src/separation/cpu_separator.py` as a fourth separation
backend (alongside `sepformer`, `channel_select`, `ica`). MVDR exploits all 4 microphone
channels spatially without requiring a GPU or ML model:
- Takes `[4, T]` input, applies delay-and-sum steering followed by MVDR weight computation.
- Returns `[n_out, T]` separated sources.
- Activated via `--sep-method beamform` or `cpu_fallback_method: beamform` in config.

### 5. Ablation Study Framework

`evaluate.py` now accepts `--sep-method [auto|channel_select|ica|beamform|sepformer]`
for direct backend comparison. `scripts/run_ablation.py` orchestrates multi-run ablation
experiments and writes consolidated CSV to `outputs/metrics/sep_backends.csv`.
`scripts/generate_tables.py` generates `outputs/tables/ablation_table.tex` from results.

### 6. Error Analysis Script

`scripts/error_analysis.py` reads WER CSV outputs and produces:
- Per-session breakdown: substitutions, insertions, deletions.
- Hallucination detection via compression ratio on hypothesis text.
- Correlation between WER and session length (proxy for silence/noise ratio).

### 7. Demo Enhancements

- `scripts/extract_demo_clips.py`: auto-selects 3–5 clips (30–60s) from `data/dev`
  with high energy variance (indicative of overlapping speech).
- `app.py`: added "Baseline vs. Pipeline Comparison" tab (`tab_compare`) showing
  side-by-side channel-0 ASR vs. separated ASR on the same clip.

### 8. SpeechBrain / Transformers API Compatibility

- **SpeechBrain ≥1.0**: `separate_batch` expects `[B, T]` input and returns `[B, T, N]` output.
  Fixed `src/separation/separator.py`.
- **SpeechBrain device string**: `"cuda"` is invalid; must use `"cuda:0"`. Fixed in
  `separator.py` and `classifier.py`.
- **Transformers 5.x**: Setting hallucination params in `generate_kwargs` crashed with
  `logprobs` scope error. Fixed by setting directly on `model.generation_config`.

---

## What Didn't Work / Remaining Challenges

### 1. Near-Field References Unavailable in Dev/Eval
AISHELL-5's near-field recordings are only in the 54 GB training split. True SI-SNRi
computation unavailable for dev/eval. Energy-ratio proxy used instead.
**Fix for Phase 3**: Download 20-session subset of train, compute proper SI-SNRi.

### 2. Speaker Role Ground Truth Missing
AISHELL-5 does not annotate Driver vs. Passenger per session. Rule-based classifier
assigns Driver to the track most correlated with channel 0 — unverifiable without labels.
**Fix for Phase 3**: Manually annotate 50 sessions.

### 3. SepFormer Domain Mismatch
SepFormer is pretrained on WSJ0-2Mix (clean, anechoic, English); applied zero-shot to
AISHELL-5 (reverberant, car-noise, Mandarin). Domain gap degrades separation quality.
**Fix for Phase 3**: Fine-tune SepFormer on AISHELL-5 train.

### 4. CER > 100% on Long Sessions (Residual)
Even with hallucination prevention, CPU session-level evaluation still produces
CER > 100% on sessions with heavy background noise and very short references.
Segment-level evaluation (`--eval-mode segment`) mitigates this but requires
per-utterance TextGrid annotations to be materialized first.

---

## Lessons Learned

1. **Transformers 5.x breaks `generate_kwargs` for Whisper** — always set whisper-specific
   params (no_speech_threshold, compression_ratio_threshold) directly on `model.generation_config`.

2. **Full-session CER is not a fair ASR metric** — it conflates hallucination artifacts with
   genuine ASR errors. Always report segment-level CER for Whisper on long-form audio.

3. **MVDR beamforming is free performance** on 4-channel recordings — it requires no training
   and exploits spatial diversity that channel_select discards.

4. **Tests on pure-Python logic are cheap** — the `test_metrics.py` and `test_intent_engine.py`
   suites run without GPU and catch regressions instantly.

5. **LaTeX tables from code, not manually** — generating `main_results.tex` and
   `ablation_table.tex` from CSV files via `generate_tables.py` eliminated copy-paste errors.

---

## Next Steps for Phase 3

1. **Download training split** and compute true SI-SNRi on 20+ sessions with near-field references.
2. **Fine-tune SepFormer** on AISHELL-5 (mixture → near-field target, 4-channel input).
3. **Annotate 50 sessions** with ground-truth Driver/Passenger labels.
4. **Whisper fine-tuning** on AISHELL-1 (clean Mandarin) or AISHELL-5 train.
5. **LLM Intent Engine**: Replace rule-based YAML with Qwen-1.5-1.8B for richer multi-turn intent.
6. **Quantization** for on-device deployment: INT8 whisper.cpp + ONNX SepFormer (target 2× latency reduction).
7. **Conference submission**: Target Interspeech 2026 with Phase 3 results (segment-level CER + ablation table).
