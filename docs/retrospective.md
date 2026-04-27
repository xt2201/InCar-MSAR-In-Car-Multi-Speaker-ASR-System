# Project Retrospective – InCar-MSAR

**Date**: 2026-04-23  
**Phase**: 1.0 (Research Pipeline + Demo)  
**Sprint Span**: S1–S10 (10 weeks)

---

## What Worked Well

### 1. Modular Architecture with Lazy Loading
The decision to organize the pipeline into discrete, independently-loadable modules
(`SpeechSeparator`, `WhisperASR`, `SpeakerRoleClassifier`, `IntentEngine`) proved
highly effective. Heavy models (Whisper, SepFormer, ECAPA) are loaded lazily,
which enables fast unit testing without GPU and allows the demo to initialize
incrementally. The `@property` pattern in `InCarASRPipeline` made this seamless.

### 2. Config-Driven Design
Externalizing all hyperparameters into `configs/default.yaml` made reproducibility
significantly easier. When comparing baseline vs. pipeline vs. demo configurations,
switching via `--config configs/demo.yaml` was frictionless. The OmegaConf library
provided dot-access and structured validation.

### 3. cpWER with Hungarian Algorithm
Implementing cpWER correctly using `scipy.optimize.linear_sum_assignment` was a
key scientific contribution. The brute-force fallback for N≤4 provides an
independent check. The toy-example unit tests (`test_metrics.py`) verified
correctness before running on real data.

### 4. Intent Engine with External YAML
Externalizing keyword mappings to `configs/intent_keywords.yaml` enabled rapid
iteration without touching Python code. Adding or modifying intent categories
requires no code changes. The 50-case labeled test dataset (`test_intent_engine.py`)
provides a regression guard.

### 5. Streamlit Demo Design
The split-panel UI (audio player on left, transcript table on right) proved
intuitive for demo purposes. Color-coding by speaker role (blue=Driver,
orange=Passenger) and intent icons (🌡️, ▶️, 🗺️) made the output immediately
readable without needing to parse JSON.

---

## What Didn't Work / Challenges

### 1. Near-Field References Unavailable in Dev/Eval
AISHELL-5's near-field (clean headset) recordings are only available in the
54 GB training split, which is too large to download in the demo environment.
This means true SI-SNRi computation (the standard separation quality metric)
is unavailable for dev/eval evaluation. We used an energy-ratio proxy, which
is less rigorous. **Fix for Phase 2**: Download train split, compute proper
SI-SNRi on a 20-session subset.

### 2. Speaker Role Ground Truth Missing
AISHELL-5 does not annotate which speaker is Driver vs. Passenger per session.
Our rule-based classifier assigns Driver to the track most correlated with
channel 0 (driver-side mic)—a reasonable assumption given microphone placement,
but not formally verifiable without annotation. **Fix for Phase 2**: Manually
annotate 50 sessions or use acoustic positioning from stereo cues.

### 3. SepFormer Domain Mismatch
SepFormer is pretrained on WSJ0-2Mix (clean, anechoic, English) and applied
zero-shot to AISHELL-5 (reverberant, car-noise, Mandarin). This domain gap
almost certainly degrades separation quality compared to what is achievable
with fine-tuning. **Fix for Phase 2**: Fine-tune SepFormer on AISHELL-5 train
(near-field as target, 4-ch mix as input).

---

## Lessons Learned

1. **Scientific integrity first**: Even when tempted to hardcode "expected" WER
   numbers to present cleaner results, maintaining placeholder outputs until
   actual experiments run was the correct approach. The evaluation framework is
   correct; numbers will follow.

2. **Fallback mechanisms are critical**: The SI-SNRi fallback (direct multi-channel
   ASR when separation quality is low) saved many demo runs from catastrophic failure.
   Engineering robustness first, then optimization.

3. **Tests on pure-Python logic are cheap**: The `test_metrics.py` and
   `test_intent_engine.py` suites run without GPU and catch regressions instantly.
   These are the highest-value tests for the project.

4. **LaTeX tables from code, not manually**: Generating `main_results.tex` and
   `ablation_table.tex` from CSV files via `generate_tables.py` eliminated
   copy-paste errors when updating numbers.

---

## Next Steps for Phase 2

1. **Download training split** and compute true SI-SNRi on 20+ sessions with
   near-field references.
2. **Fine-tune SepFormer** on AISHELL-5 (mixture → near-field target, 4-channel
   input) to close domain gap.
3. **Annotate 50 sessions** with ground-truth Driver/Passenger labels for proper
   Speaker Accuracy evaluation.
4. **Whisper fine-tuning** on AISHELL-1 (clean Mandarin) or AISHELL-5 train to
   reduce ASR WER for in-car far-field audio.
5. **LLM Intent Engine**: Replace rule-based YAML with Qwen-1.5-1.8B for richer
   multi-turn intent understanding.
6. **Quantization** for on-device deployment: INT8 whisper.cpp + ONNX SepFormer
   targeting 2x latency reduction.
7. **Conference submission**: Target Interspeech 2026 with Phase 2 results.
