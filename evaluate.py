#!/usr/bin/env python3
"""Full evaluation script for In-Car Multi-Speaker ASR Pipeline.

Usage:
    python evaluate.py --split eval1 --n 100
    python evaluate.py --split dev --n 50 --config configs/default.yaml
    python evaluate.py --split eval2 --mode baseline  # single-channel baseline

Ablation study (separation backend):
    python evaluate.py --split eval1 --mode pipeline --sep-method channel_select
    python evaluate.py --split eval1 --mode pipeline --sep-method beamform
    python evaluate.py --split eval1 --mode pipeline --sep-method sepformer

Segment-level evaluation (per-utterance, avoids Whisper hallucination on long sessions):
    # First export segments: python scripts/materialize_aishell5_flat.py ... --export-segments
    python evaluate.py --split eval1 --mode pipeline --eval-mode segment
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from loguru import logger

from src.pipeline.orchestrator import InCarASRPipeline
from src.pipeline.data_loader import AISHELL5Loader
from src.evaluation.metrics import (
    assign_references_to_hypotheses,
    compute_wer,
    compute_cpwer,
    summarize_metrics,
)
from src.utils.config import load_config

# Anchor relative paths in config/CLI to the repository root (directory of this file),
# not the process CWD — reliable on WSL/CI and when invoking from another directory.
REPO_ROOT = Path(__file__).resolve().parent

# Valid separation method choices for ablation
_SEP_METHODS = ["auto", "sepformer", "channel_select", "ica", "beamform"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate InCar ASR Pipeline")
    p.add_argument("--split", choices=["dev", "eval1", "eval2"], default="eval1",
                   help="Dataset split to evaluate on")
    p.add_argument("--n", type=int, default=-1,
                   help="Number of samples to evaluate (-1 = all)")
    p.add_argument("--config", default="configs/default.yaml",
                   help="Path to config YAML")
    p.add_argument("--mode", choices=["pipeline", "baseline", "upper_bound"],
                   default="pipeline",
                   help="Evaluation mode: full pipeline, single-channel baseline, or near-field upper-bound")
    p.add_argument(
        "--sep-method",
        choices=_SEP_METHODS,
        default="auto",
        help=(
            "Override separation backend for ablation study. "
            "'auto' uses config default. "
            "Options: sepformer (GPU), channel_select (CPU), ica (CPU), beamform (CPU MVDR)."
        ),
    )
    p.add_argument(
        "--eval-mode",
        choices=["session", "segment"],
        default="session",
        help=(
            "Evaluation granularity. 'session' evaluates full sessions (legacy, "
            "prone to Whisper hallucination on long audio). 'segment' evaluates "
            "per-utterance clips from data/{split}/segments/ (requires --export-segments "
            "in materialize_aishell5_flat.py)."
        ),
    )
    p.add_argument("--output-dir", default="outputs/metrics",
                   help="Directory to save output CSV files")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--drop-latency-warmup",
        type=int,
        default=1,
        help="Drop first N sessions from latency mean/P95 (model cold start). 0=keep all.",
    )
    return p.parse_args()


def run_baseline(sample, asr) -> list[dict]:
    """Single-channel baseline: run Whisper on channel 0 only (no separation)."""
    from src.utils.audio import load_multichannel_audio
    waveform, sr = load_multichannel_audio(sample.wav_path)
    channel0 = waveform[0]  # [T]
    result = asr.transcribe(channel0, sample_rate=sr)
    return [{
        "file_id": sample.session_id,
        "speaker_id": 0,
        "role": "Driver",
        "transcript": result["text"],
        "asr_ms": result["inference_time_ms"],
    }]


def n_speakers_for_sample(sample) -> int | None:
    """Infer separation count from ground-truth references (2–4 per AISHELL-5)."""
    r = len(sample.references) if sample.references else 0
    if r == 0:
        return None
    if r == 1:
        return 2
    return min(max(r, 2), 4)


def run_upper_bound(sample, asr) -> list[dict]:
    """Upper-bound: run Whisper on each channel independently (best-case far-field).

    Since near-field (headset) data is only available in AISHELL-5 train split,
    this mode uses per-channel far-field ASR as a proxy upper bound.
    Each channel is treated as one speaker source.
    """
    from src.utils.audio import load_multichannel_audio
    waveform, sr = load_multichannel_audio(sample.wav_path)
    nref = len(sample.references) if sample.references else 4
    n_ch = min(int(waveform.shape[0]), max(nref, 1))
    utterances = []
    for ch in range(n_ch):
        result = asr.transcribe(waveform[ch], sample_rate=sr)
        utterances.append({
            "file_id": sample.session_id,
            "speaker_id": ch,
            "role": f"Channel_{ch}",
            "transcript": result["text"],
            "asr_ms": result["inference_time_ms"],
        })
    return utterances


def apply_sep_method_override(pipeline: "InCarASRPipeline", sep_method: str) -> None:
    """Override the separation backend on a pipeline for ablation studies.

    This patches the pipeline's config at runtime so that the next call to
    ``pipeline.separator`` (lazy property) picks up the desired backend.
    Does nothing when sep_method is "auto" (use config default).
    """
    if sep_method == "auto":
        return
    # Map CLI name → config cpu_fallback_method or force sepformer
    cfg = pipeline.cfg.separation
    if sep_method == "sepformer":
        import torch
        if not torch.cuda.is_available():
            logger.warning("--sep-method sepformer requested but no GPU; using channel_select instead.")
            sep_method = "channel_select"
        else:
            # Force auto→cuda so SepFormer is selected
            cfg.device = "cuda"
            pipeline._chunked_sep = None  # invalidate cached separator
            return
    # CPU methods: set device=cpu and cpu_fallback_method
    cfg.device = "cpu"
    cfg.cpu_fallback_method = sep_method  # "channel_select", "ica", or "beamform"
    pipeline._chunked_sep = None  # invalidate cached separator
    logger.info(f"Separation backend overridden → {sep_method}")


def run_segment_eval(data_dir: Path, split_id: str, asr) -> list[dict]:
    """Evaluate on per-utterance segments (requires --export-segments materialization).

    Reads from data/{split}/segments/{session_id}/refs.json and the
    corresponding WAV clips, runs ASR on each clip, returns per-utterance rows.

    Returns a list of dicts with keys: file_id, session_id, speaker_id, role,
    transcript, reference, asr_ms, clip_dur_s.
    """
    import soundfile as sf

    segments_root = data_dir / "segments"
    if not segments_root.is_dir():
        logger.error(
            f"Segment directory not found: {segments_root}\n"
            "Run: python scripts/materialize_aishell5_flat.py ... --export-segments"
        )
        return []

    rows: list[dict] = []
    session_dirs = sorted(segments_root.iterdir())
    for sess_dir in session_dirs:
        if not sess_dir.is_dir():
            continue
        refs_path = sess_dir / "refs.json"
        if not refs_path.exists():
            continue
        refs_data: dict = json.loads(refs_path.read_text(encoding="utf-8"))
        session_id = sess_dir.name

        for spk_id, segs in refs_data.items():
            spk_num = int("".join(c for c in spk_id if c.isdigit()) or "0")
            for seg in segs:
                clip_path = sess_dir / seg["clip"]
                if not clip_path.exists():
                    logger.warning(f"Missing clip: {clip_path}")
                    continue
                try:
                    data_np, sr = sf.read(str(clip_path))
                    if data_np.ndim == 2:
                        data_np = data_np[:, 0]
                    import torch
                    clip_tensor = torch.tensor(data_np, dtype=torch.float32)
                    result = asr.transcribe(clip_tensor, sample_rate=sr)
                    rows.append({
                        "file_id": seg["clip"],
                        "session_id": session_id,
                        "speaker_id": spk_num,
                        "role": "Driver" if spk_num == 1 else f"Passenger_{spk_num - 1}",
                        "transcript": result["text"],
                        "reference": seg["text"],
                        "asr_ms": round(result["inference_time_ms"], 1),
                        "is_silent": result.get("is_silent", False),
                        "clip_dur_s": round(seg["end"] - seg["start"], 2),
                    })
                except Exception as e:
                    logger.error(f"Error on {clip_path}: {e}")

    return rows


def evaluate(args: argparse.Namespace) -> None:
    import random
    random.seed(args.seed)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    cfg = load_config(config_path)

    # Resolve data directory (relative to repo root, not CWD)
    data_root = Path(cfg.paths.data_root)
    if not data_root.is_absolute():
        data_root = (REPO_ROOT / data_root).resolve()
    data_dir = (data_root / args.split).resolve()
    if not data_dir.is_dir():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info(f"Run: bash scripts/download_data.sh --{args.split}")
        raise SystemExit(1)

    # Initialize pipeline
    pipeline = InCarASRPipeline(str(config_path))

    # Apply separation method override for ablation
    sep_method = getattr(args, "sep_method", "auto")
    if sep_method and sep_method != "auto":
        apply_sep_method_override(pipeline, sep_method)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build output filename (include sep_method suffix for ablation runs)
    sep_suffix = f"_{sep_method}" if sep_method and sep_method != "auto" else ""
    eval_mode = getattr(args, "eval_mode", "session")
    eval_suffix = "_seg" if eval_mode == "segment" else ""
    output_csv = output_dir / f"wer_{args.mode}{sep_suffix}{eval_suffix}_{args.split}.csv"

    wer_list = []
    cpwer_list = []
    latency_list = []
    results_rows = []

    # -----------------------------------------------------------------------
    # Segment-level evaluation path
    # -----------------------------------------------------------------------
    if eval_mode == "segment":
        logger.info(f"Segment-level evaluation on {args.split}")
        t0_all = time.perf_counter()
        seg_rows = run_segment_eval(data_dir, args.split, pipeline.asr)

        if not seg_rows:
            logger.warning("No segment rows produced. Check --export-segments materialization.")
        else:
            for row in seg_rows:
                hyp = row["transcript"]
                ref = row.get("reference", "")
                wer = compute_wer(hyp, ref) if ref and not row.get("is_silent") else float("nan")
                results_rows.append({
                    "file_id": row["file_id"],
                    "session_id": row.get("session_id", ""),
                    "speaker_id": row.get("speaker_id", 0),
                    "role": row.get("role", "unknown"),
                    "hypothesis": hyp,
                    "reference": ref,
                    "wer": round(wer, 4),
                    "asr_ms": row.get("asr_ms", 0),
                    "clip_dur_s": row.get("clip_dur_s", 0),
                })
                if not float_isnan_safe(wer):
                    wer_list.append(wer)
            latency_list.append((time.perf_counter() - t0_all) * 1000)

        # cpWER not applicable per-segment (no multi-speaker permutation per clip)
        logger.info(
            f"Segment eval: {len(results_rows)} clips, "
            f"mean CER={float(np.mean(wer_list)):.1%} (n={len(wer_list)})"
            if wer_list else "Segment eval: 0 valid clips"
        )

    else:
        # -----------------------------------------------------------------------
        # Session-level evaluation path (legacy + pipeline mode)
        # -----------------------------------------------------------------------
        max_n = args.n if args.n > 0 else None
        loader = AISHELL5Loader(data_dir, max_samples=max_n)
        logger.info(f"Evaluating on {len(loader)} samples from {args.split} set ({eval_mode})")

        for i, sample in enumerate(loader):
            logger.info(f"[{i+1}/{len(loader)}] {sample.session_id}")

            try:
                t0 = time.perf_counter()

                n_spk = n_speakers_for_sample(sample)
                if args.mode == "baseline":
                    utterances = run_baseline(sample, pipeline.asr)
                elif args.mode == "upper_bound":
                    utterances = run_upper_bound(sample, pipeline.asr)
                else:
                    utterances = pipeline.process_file(sample.wav_path, n_speakers=n_spk)

                elapsed_ms = (time.perf_counter() - t0) * 1000

                # SPK1, SPK2, … (stable) — *not* the same as separated-track order
                ref_texts = [sample.references[k] for k in sorted(sample.references.keys())] if sample.references else []
                hyp_nonsilent = [u["transcript"] for u in utterances if not u.get("is_silent")]
                hyp_texts = list(hyp_nonsilent)
                if ref_texts and hyp_nonsilent:
                    matched = assign_references_to_hypotheses(ref_texts, hyp_nonsilent)
                else:
                    matched = [""] * len(hyp_nonsilent)
                m_idx = 0

                for j, utt in enumerate(utterances):
                    hyp = utt["transcript"]
                    if utt.get("is_silent"):
                        ref = ""
                    else:
                        ref = matched[m_idx] if m_idx < len(matched) else ""
                        m_idx += 1
                    wer = compute_wer(hyp, ref) if ref else float("nan")

                    results_rows.append({
                        "file_id": sample.session_id,
                        "speaker_id": utt.get("speaker_id", j),
                        "role": utt.get("role", "unknown"),
                        "hypothesis": hyp,
                        "reference": ref,
                        "wer": round(wer, 4),
                        "asr_ms": round(utt.get("asr_ms", 0), 1),
                        "total_ms": round(elapsed_ms, 1),
                        "sep_method": sep_method,
                    })

                    if not float_isnan_safe(wer):
                        wer_list.append(wer)

                # cpWER
                if ref_texts and hyp_texts:
                    cpwer = compute_cpwer(hyp_texts, ref_texts)
                    cpwer_list.append(cpwer)

                latency_list.append(elapsed_ms)

            except Exception as e:
                logger.error(f"Error processing {sample.session_id}: {e}")
                results_rows.append({
                    "file_id": sample.session_id,
                    "speaker_id": -1,
                    "role": "error",
                    "hypothesis": "",
                    "reference": "",
                    "wer": float("nan"),
                    "asr_ms": 0,
                    "total_ms": 0,
                    "sep_method": sep_method,
                    "error": str(e),
                })

    # Write per-sample results
    if results_rows:
        # Unify fieldnames across all rows (error rows may have extra keys)
        fieldnames_set: set[str] = set()
        for row in results_rows:
            fieldnames_set.update(row.keys())
        fieldnames = sorted(fieldnames_set)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in results_rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        logger.info(f"Results saved to {output_csv}")

    # Summary statistics
    summary = summarize_metrics(wer_list, cpwer_list)
    # Latency: optional warmup drop (avoids one cold start skewing mean vs P95)
    lat_for_stats = latency_list
    if args.drop_latency_warmup > 0 and len(latency_list) > args.drop_latency_warmup:
        lat_for_stats = latency_list[args.drop_latency_warmup :]
    arr = np.array(lat_for_stats, dtype=np.float64) if lat_for_stats else np.array([], dtype=np.float64)
    p95 = float(np.percentile(arr, 95)) if len(arr) else 0.0
    summary["latency"] = {
        "mean_ms": round(float(np.mean(arr)), 1) if len(arr) else 0.0,
        "p95_ms": round(p95, 1) if len(arr) else 0.0,
        "median_ms": round(float(np.median(arr)), 1) if len(arr) else 0.0,
        "n": len(latency_list),
        "n_for_latency_stats": len(arr),
        "dropped_warmup": int(len(latency_list) - len(arr)) if latency_list else 0,
    }
    summary["mode"] = args.mode
    summary["split"] = args.split
    summary["sep_method"] = sep_method
    summary["eval_mode"] = eval_mode
    if eval_mode != "segment":
        try:
            n_ld = len(loader)  # type: ignore[name-defined]
            any_synth = (
                all(str(loader[i].session_id).startswith("synth_") for i in range(n_ld))  # type: ignore[name-defined]
                if n_ld else False
            )
            summary["n_samples"] = n_ld
        except NameError:
            any_synth = False
            summary["n_samples"] = len(results_rows)
    else:
        summary["n_samples"] = len(results_rows)
        any_synth = False
    summary["data_note"] = "synthetic_sine_mixture" if any_synth else "real_aishell5"

    summary_path = output_dir / f"summary_{args.mode}{sep_suffix}{eval_suffix}_{args.split}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info(
        f"EVALUATION SUMMARY: mode={args.mode}, split={args.split}, "
        f"sep={sep_method}, eval_mode={eval_mode}"
    )
    logger.info(f"  WER:   mean={summary['wer']['mean']:.1%}  (n={summary['wer']['n']})")
    logger.info(f"  cpWER: mean={summary['cpwer']['mean']:.1%} (n={summary['cpwer']['n']})")
    logger.info(
        f"  Latency: mean={summary['latency']['mean_ms']:.0f}ms, "
        f"median={summary['latency']['median_ms']:.0f}ms, P95={summary['latency']['p95_ms']:.0f}ms"
    )
    logger.info(f"  Data: {summary.get('data_note', 'unknown')}")
    logger.info(f"  Summary: {summary_path}")
    logger.info("=" * 60)


def float_isnan_safe(v) -> bool:
    """Return True if v is NaN/inf/invalid (skip in WER aggregation)."""
    if v is None:
        return True
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return True
    return not math.isfinite(f)


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
