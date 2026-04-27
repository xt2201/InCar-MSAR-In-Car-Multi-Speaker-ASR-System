# InCar-MSAR: System Architecture & Diagrams

> **Project**: In-Car Multi-Speaker Automatic Speech Recognition  
> **Dataset**: AISHELL-5 (4-channel, Mandarin Chinese)  
> **Stack**: Python 3.10 · PyTorch · SpeechBrain · OpenAI Whisper · Asteroid · Streamlit  
> **Last updated**: 2026-04-27 (Phase 2 improvements)

---

## Architecture Summary

**InCar-MSAR** is an end-to-end pipeline for recognizing simultaneous speech from multiple passengers in a vehicle cabin. The system ingests 4-channel far-field microphone audio and produces per-speaker structured outputs containing the transcript, speaker role (Driver / Passenger), and parsed in-car command intent.

The architecture follows a **sequential modular pipeline** pattern with lazy-loaded ML models, adaptive device routing (GPU ↔ CPU fallback), and a unified evaluation harness. All modules communicate via typed Python dicts, making each stage independently testable and replaceable.

Key design decisions:
- **Lazy instantiation** — heavy models (Whisper, SepFormer, ECAPA-TDNN) are loaded on first use, enabling fast startup for Streamlit demos.
- **Device-aware routing** — a `get_separator()` factory dispatches to `SepFormer` (GPU), `ChannelSelectSeparator`/`ICASeparator` (CPU), or `BeamformSeparator` (MVDR, CPU multi-channel) at runtime.
- **Fallback logic** — if separation quality (SI-SNRi) is too low, the pipeline bypasses separation and feeds raw microphone channels directly to ASR.
- **Hybrid speaker classification** — ECAPA-TDNN embeddings verify or override the rule-based energy heuristic; if ECAPA exceeds its latency budget, it falls back to rule-based.
- **ECAPA device** — `ECAPAEmbedding` uses the **same resolved device as Whisper** (`_asr_device` in `InCarASRPipeline`), not `separation.device`, so both track GPU or CPU coherently.
- **Per-file speaker count** — `InCarASRPipeline.process_file(..., n_speakers=...)` can override the default `n_speakers` for AISHELL-5 sessions with 2–4 reference speakers in the transcript.
- **Dataset splits** — see `docs/dataset.md` §9: **dev, eval1, eval2** use materialized `wav/` + `text/`; **noise** is environmental-only and is **not** a WER split in `evaluate.py`.
- **Hallucination prevention** — `WhisperASR` applies `no_speech_threshold`, `compression_ratio_threshold`, and delegates audio >30s to `ChunkedASR` to prevent repetition artifacts on long silences.
- **Segment-level evaluation** — `evaluate.py --eval-mode segment` cuts sessions into per-utterance clips (from TextGrid) before feeding Whisper, producing accurate WER without hallucination bias.
- **Ablation-ready CLI** — `evaluate.py --sep-method [auto|channel_select|ica|beamform|sepformer]` allows direct backend comparison without editing config files.

## Implementation order vs. diagrams

`InCarASRPipeline` implements this **strict sequence** (CPU or GPU):

1. `load_multichannel_audio` → 4-ch tensor `[4, T]`.
2. `ChunkedSeparator.process_file` → separated sources `[N, T]` (input is mono-mixed for SepFormer, full multichannel for CPU backends).
3. Optional quality fallback → replace sources with per-mic slices if the heuristic says separation failed (GPU / neural path; CPU channel-select is usually skipped for this check).
4. `SpeakerRoleClassifier` / `RuleBasedRoleClassifier` on **(sources, mixture)** → `role_map`.
5. `WhisperASR.transcribe` per source track → `IntentEngine.parse` per row.

Mermaid *Diagram 1* draws parallel edges from `SEP` to `SPK` and `ASR` for data dependency; the **code runs** steps 4 then 5 so that **role is known before** intent parsing (`IntentEngine` receives `speaker=role`). This matches the “separation → role → ASR+intent” reading of the product spec.

## Evaluation (AISHELL-5)

- **cpWER** is the primary multi-speaker metric; Hungarian assignment matches hypotheses to `SPK1, SPK2, …` references.
- **Per-utterance WER in CSV** uses the **same** optimal assignment as cpWER via `assign_references_to_hypotheses()` so row-level WER is not a misleading by-product of track order.

---

## Key Components

| Module | Class(es) | Responsibility |
|---|---|---|
| `src/pipeline/orchestrator.py` | `InCarASRPipeline` | End-to-end orchestration; lazy model loading; streaming/file-level API |
| `src/pipeline/data_loader.py` | `AISHELL5Loader`, `Sample` | AISHELL-5 WAV + transcript loading and parsing |
| `src/separation/separator.py` | `SpeechSeparator`, `ChunkedSeparator` | SepFormer / ConvTasNet inference; overlap-add chunked streaming |
| `src/separation/cpu_separator.py` | `ChannelSelectSeparator`, `ICASeparator`, `BeamformSeparator`, `get_separator()` | CPU-friendly separation; MVDR beamforming on 4-ch input; factory dispatch |
| `src/asr/whisper_asr.py` | `WhisperASR`, `ChunkedASR` | HuggingFace Whisper pipeline; silence + hallucination detection; auto-chunks audio >30s |
| `src/speaker/classifier.py` | `RuleBasedRoleClassifier`, `ECAPAEmbedding`, `SpeakerRoleClassifier` | Driver/Passenger role attribution via energy correlation and 192-dim embeddings |
| `src/intent/engine.py` | `IntentEngine` | YAML-driven keyword matching → structured intent JSON (12 categories) |
| `src/evaluation/metrics.py` | standalone functions | WER (CER), cpWER (Hungarian), Speaker Accuracy, SI-SNRi |
| `configs/default.yaml` | — | Single config file: audio, separation, ASR (whisper-small, beam=2), speaker, intent, evaluation, paths |
| `evaluate.py` | `evaluate()` | Batch evaluation harness: baseline / pipeline / upper-bound; `--sep-method` ablation; `--eval-mode session|segment` |
| `scripts/error_analysis.py` | standalone script | Breakdown WER into substitution/insertion/deletion; detect hallucination via compression ratio |
| `scripts/extract_demo_clips.py` | standalone script | Auto-select 30–60s demo clips with audible overlap from data/dev |
| `app.py` | Streamlit app | Real-time demo + baseline vs. pipeline side-by-side comparison tab |

---

## Pipelines & Workflows

Three distinct execution paths exist:

1. **Inference Pipeline** (`InCarASRPipeline.process_file`) — Production path. Loads WAV → Separates → Classifies speakers → ASR per track → Parses intent.
2. **Streaming Pipeline** (`InCarASRPipeline.process_chunk`) — Real-time path. Processes a 2-second audio chunk without reading a full file.
3. **Evaluation Harness** (`evaluate.py`) — Batch path. Iterates over `AISHELL5Loader`, runs inference (pipeline/baseline/upper_bound), computes WER/cpWER/latency, saves CSV + JSON summary.

---

## Mermaid Diagrams

### Diagram 1 — High-Level System Architecture

This diagram shows the three top-level system boundaries: the data ingestion layer (AISHELL-5), the core inference pipeline, and the evaluation/demo output layer. Dashed lines indicate optional or conditional paths.

```mermaid
graph TB
    subgraph DATA["Data Layer"]
        A5["AISHELL-5 Dataset<br/>4-ch WAV + Transcripts"]
        CFG["configs/default.yaml<br/>Audio · Sep · ASR · Speaker · Intent"]
    end

    subgraph PIPELINE["InCarASRPipeline (Orchestrator)"]
        SEP["Speech Separation<br/>SepFormer / ChannelSelect / ICA"]
        SPK["Speaker Role Classifier<br/>Rule-Based + ECAPA-TDNN"]
        ASR["Whisper ASR<br/>openai/whisper-tiny | small"]
        INT["Intent Engine<br/>Keyword Matching (12 categories)"]
    end

    subgraph OUTPUT["Output Layer"]
        UTT["Utterance JSON<br/>role · transcript · intent · latency_ms"]
        EVAL["Evaluation Harness<br/>WER · cpWER · Speaker Acc · Latency"]
        DEMO["Streamlit Demo<br/>app.py"]
        METRICS["outputs/metrics/<br/>*.csv · *.json"]
    end

    A5 -->|"WAV [4, T] @ 16kHz"| SEP
    CFG -->|"config object"| PIPELINE
    SEP -->|"sources [N, T]"| SPK
    SEP -->|"sources [N, T]"| ASR
    SPK -->|"role_map {spk_id → role}"| UTT
    ASR -->|"transcript, is_silent"| INT
    INT -->|"intent JSON"| UTT
    UTT --> EVAL
    UTT --> DEMO
    EVAL --> METRICS
```

---

### Diagram 2 — Component & Module Diagram

This diagram details the class-level structure within each module, showing inheritance, composition, and the factory dispatch pattern for separation backends.

```mermaid
graph LR
    subgraph orchestrator["pipeline/orchestrator.py"]
        ORC["InCarASRPipeline<br/>─────────────────<br/>+ process_file()<br/>+ process_chunk()<br/>+ benchmark_memory()<br/>─ separator [lazy]<br/>─ asr [lazy]<br/>─ speaker_classifier [lazy]<br/>─ intent_engine [lazy]"]
    end

    subgraph sep_mod["separation/"]
        SSEP["SpeechSeparator<br/>─────────────<br/>+ separate()<br/>+ compute_si_snri()<br/>[SepFormer | ConvTasNet]"]
        CSEP["ChunkedSeparator<br/>─────────────<br/>+ process_file()<br/>─ _overlap_add()"]
        CHSEL["ChannelSelectSeparator<br/>─────────────<br/>+ separate()<br/>[RMS energy sort]"]
        ICA["ICASeparator<br/>─────────────<br/>+ separate()<br/>[FastICA / sklearn]"]
        FAC["get_separator()<br/>Factory Function<br/>[GPU→SepFormer<br/>CPU→ChannelSelect | ICA]"]

        FAC --> SSEP
        FAC --> CHSEL
        FAC --> ICA
        CSEP -->|wraps| SSEP
    end

    subgraph asr_mod["asr/"]
        WASR["WhisperASR<br/>─────────────<br/>+ transcribe()<br/>+ transcribe_batch()<br/>[HuggingFace pipeline]"]
        CASR["ChunkedASR<br/>─────────────<br/>+ transcribe_long()<br/>[wraps WhisperASR]"]
        CASR -->|wraps| WASR
    end

    subgraph spk_mod["speaker/"]
        RBC["RuleBasedRoleClassifier<br/>─────────────<br/>+ classify()<br/>+ classify_by_energy()<br/>[channel correlation]"]
        ECAPA["ECAPAEmbedding<br/>─────────────<br/>+ extract() → [1,192]<br/>+ cosine_similarity()<br/>[SpeechBrain ECAPA-TDNN]"]
        SRC["SpeakerRoleClassifier<br/>─────────────<br/>+ classify()<br/>[hybrid: ECAPA + rule]"]

        SRC -->|fallback| RBC
        SRC -->|uses| ECAPA
    end

    subgraph intent_mod["intent/"]
        IE["IntentEngine<br/>─────────────<br/>+ parse()<br/>+ parse_batch()<br/>+ supported_intents()<br/>[YAML keyword rules]"]
    end

    subgraph data_mod["pipeline/data_loader.py"]
        LDR["AISHELL5Loader<br/>─────────────<br/>+ __iter__()<br/>+ statistics()"]
        SMP["Sample<br/>─────────────<br/>session_id, wav_path<br/>references, n_speakers"]
        LDR -->|yields| SMP
    end

    ORC -->|"lazy property"| FAC
    ORC -->|"lazy property"| WASR
    ORC -->|"lazy property"| SRC
    ORC -->|"lazy property"| IE
```

---

### Diagram 3 — Data Flow / Processing Pipeline

This diagram traces the transformation of raw audio data from disk through each processing stage to the final structured utterance output. Tensor shapes and data types are annotated at key edges.

```mermaid
flowchart LR
    WAV["WAV File<br/>4-ch · 16kHz<br/>shape: [4, T]"]

    MIX["Channel Mix<br/>mono [1, T]<br/>(SepFormer path only)"]

    CHUNK["ChunkedSeparator<br/>2s chunks<br/>+ 0.2s overlap"]

    SEP_OUT["Separated Sources<br/>shape: [N, T]<br/>N = 2–4 speakers"]

    QUALITY{{"SI-SNRi Check<br/>energy ratio > 1.1?"}}

    FALLBACK["Fallback Sources<br/>raw mic channels<br/>[N, T] from [4, T]"]

    ROLE["RuleBasedRoleClassifier<br/>cross-correlation<br/>with driver channel [0]"]

    ECAPA_EMB["ECAPA-TDNN<br/>192-dim embedding<br/>per track"]

    ROLE_MAP["role_map<br/>{0: 'Driver', 1: 'Passenger_1'}"]

    WHISPER["WhisperASR<br/>HuggingFace pipeline<br/>lang=zh, beam=1 | 2"]

    SILENCE{{"RMS < 1e-4?<br/>(silence check)"}}

    TRANSCRIPT["transcript: str<br/>is_silent: bool<br/>inference_ms: float"]

    INTENT["IntentEngine<br/>keyword regex match<br/>12 intent categories"]

    UTTERANCE["Utterance Dict<br/>──────────────<br/>file_id · timestamp<br/>speaker_id · role<br/>transcript · intent<br/>sep_ms · asr_ms · total_ms"]

    WAV -->|"load_multichannel_audio()"| MIX
    WAV --> CHUNK
    MIX --> CHUNK
    CHUNK -->|"overlap-add"| SEP_OUT
    SEP_OUT --> QUALITY

    QUALITY -->|"yes – good separation"| ROLE
    QUALITY -->|"no – low SI-SNRi"| FALLBACK

    FALLBACK --> ROLE
    WAV -->|"multichannel ref"| ROLE

    ROLE -->|"rule-based"| ROLE_MAP
    ROLE --> ECAPA_EMB
    ECAPA_EMB -->|"cosine verify / override"| ROLE_MAP

    SEP_OUT --> WHISPER
    FALLBACK --> WHISPER

    WHISPER --> SILENCE
    SILENCE -->|"silent"| TRANSCRIPT
    SILENCE -->|"speech"| TRANSCRIPT

    TRANSCRIPT --> INTENT

    ROLE_MAP --> UTTERANCE
    TRANSCRIPT --> UTTERANCE
    INTENT --> UTTERANCE
```

---

### Diagram 4 — Evaluation Workflow / Sequence Diagram

This sequence diagram shows the interaction between the evaluation harness, data loader, and pipeline across a single evaluation session, covering all three evaluation modes.

```mermaid
sequenceDiagram
    actor User
    participant CLI as "evaluate.py (CLI)"
    participant CFG as "load_config()"
    participant LDR as "AISHELL5Loader"
    participant PL as "InCarASRPipeline"
    participant MET as "metrics.py"
    participant FS as "outputs/metrics/"

    User->>CLI: python evaluate.py --split eval1 --mode pipeline --n 100
    CLI->>CFG: load_config("configs/default.yaml")
    CFG-->>CLI: config object
    CLI->>LDR: AISHELL5Loader(data/eval1, max_samples=100)
    LDR-->>CLI: 100 × Sample(wav_path, references)

    loop For each Sample
        CLI->>PL: process_file(sample.wav_path)
        Note over PL: Stage 1 – load_multichannel_audio()
        Note over PL: Stage 2 – ChunkedSeparator.process_file()
        Note over PL: Stage 3 – _classify_speakers()
        Note over PL: Stage 4 – WhisperASR.transcribe() × N
        Note over PL: Stage 5 – IntentEngine.parse() × N
        PL-->>CLI: utterances [list[dict]]

        CLI->>MET: compute_wer(hyp, ref) per utterance
        MET-->>CLI: wer: float

        CLI->>MET: compute_cpwer(hyp_texts, ref_texts)
        MET-->>MET: Hungarian assignment (scipy) or brute-force
        MET-->>CLI: cpwer: float
    end

    CLI->>MET: summarize_metrics(wer_list, cpwer_list)
    MET-->>CLI: summary {mean, median, std, P95}

    CLI->>FS: write wer_pipeline_eval1.csv
    CLI->>FS: write summary_pipeline_eval1.json
    CLI-->>User: print summary to stdout
```

---

### Diagram 5 — Device & Separation Backend Selection (Decision Tree)

This diagram captures the implicit runtime branching logic for selecting the appropriate separation and speaker classification backend based on hardware availability and configuration.

```mermaid
flowchart TD
    START(["InCarASRPipeline.separator<br/>(lazy property accessed)"])

    CFG_DEV{{"config.separation.device<br/>= 'auto' | 'cuda' | 'cpu'"}}

    GPU_AVAIL{{"torch.cuda.is_available()"}}

    SEPFORMER["SepFormer<br/>(SpeechBrain GPU)<br/>High-quality neural separation"]

    CPU_METHOD{{"cpu_fallback_method<br/>= 'channel_select' | 'ica' | 'beamform'"}}

    CHAN_SEL["ChannelSelectSeparator<br/>Zero-ML, instant<br/>Top-N energy channels"]

    ICA_SEP["ICASeparator<br/>FastICA (sklearn)<br/>Blind source separation"]

    BEAM_SEP["BeamformSeparator<br/>MVDR Beamforming (CPU)<br/>4-channel spatial filtering"]

    ICA_AVAIL{{"sklearn available?"}}

    CHAN_FALLBACK["ChannelSelectSeparator<br/>(hard fallback)"]

    START --> CFG_DEV
    CFG_DEV -->|"auto"| GPU_AVAIL
    CFG_DEV -->|"cuda"| GPU_AVAIL
    CFG_DEV -->|"cpu"| CPU_METHOD
    GPU_AVAIL -->|"yes"| SEPFORMER
    GPU_AVAIL -->|"no"| CPU_METHOD
    CPU_METHOD -->|"channel_select"| CHAN_SEL
    CPU_METHOD -->|"ica"| ICA_AVAIL
    CPU_METHOD -->|"beamform"| BEAM_SEP
    ICA_AVAIL -->|"yes"| ICA_SEP
    ICA_AVAIL -->|"no"| CHAN_FALLBACK
```
