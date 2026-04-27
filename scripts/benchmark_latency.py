#!/usr/bin/env python3
"""Benchmark end-to-end latency of the pipeline.

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
Measures per-chunk latency for separation and ASR separately.

Usage:
    python benchmark_latency.py --n-chunks 50 --model-size small
    python benchmark_latency.py --compare-models  # Compare tiny/base/small/medium
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from loguru import logger

from src.asr.whisper_asr import WhisperASR
from src.utils.audio import load_multichannel_audio, mix_channels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Latency benchmark")
    p.add_argument("--n-chunks", type=int, default=50)
    p.add_argument("--model-size", default="small",
                   choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"])
    p.add_argument("--chunk-sec", type=float, default=2.0)
    p.add_argument("--beam-size", type=int, default=2)
    p.add_argument("--data-dir", default="data/dev")
    p.add_argument("--output-dir", default="outputs/metrics")
    p.add_argument("--compare-models", action="store_true",
                   help="Benchmark all model sizes and save comparison table")
    return p.parse_args()


def benchmark_single_model(
    model_size: str,
    data_dir: Path,
    n_chunks: int,
    chunk_sec: float,
    beam_size: int,
) -> list[dict]:
    """Benchmark a single Whisper model size."""
    model_id = f"openai/whisper-{model_size}"
    logger.info(f"Benchmarking: {model_id} (beam_size={beam_size})")

    from src.separation.cpu_separator import get_separator
    separator = get_separator(method="auto", n_speakers=2)
    asr = WhisperASR(model_id=model_id, language="zh", device="auto", beam_size=beam_size)

    wav_files = sorted(data_dir.glob("wav/*.wav"))
    if not wav_files:
        logger.error(f"No WAV files found in {data_dir}/wav/")
        return []

    rows = []
    chunk_count = 0
    warmup_done = False

    for wav_path in wav_files:
        if chunk_count >= n_chunks:
            break
        try:
            waveform, sr = load_multichannel_audio(wav_path)
            chunk_samples = int(chunk_sec * sr)

            for i in range(0, waveform.shape[-1], chunk_samples):
                if chunk_count >= n_chunks:
                    break

                mc_chunk = waveform[:, i : i + chunk_samples]
                if mc_chunk.shape[-1] < chunk_samples:
                    mc_chunk = torch.nn.functional.pad(mc_chunk, (0, chunk_samples - mc_chunk.shape[-1]))

                # Warm-up: run one forward pass without recording
                if not warmup_done:
                    separator.separate(mc_chunk)
                    mono_wu = mix_channels(mc_chunk).squeeze()
                    asr.transcribe(mono_wu, sample_rate=sr)
                    warmup_done = True
                    logger.info("  Warm-up pass done (not timed).")

                # Separation timing
                t0 = time.perf_counter()
                sep_result = separator.separate(mc_chunk)
                sep_ms = (time.perf_counter() - t0) * 1000
                sources = sep_result["sources"]

                # ASR timing (per speaker track, sequential)
                asr_ms_total = 0.0
                for src in sources:
                    t1 = time.perf_counter()
                    asr.transcribe(src, sample_rate=sr)
                    asr_ms_total += (time.perf_counter() - t1) * 1000

                total_ms = sep_ms + asr_ms_total

                rows.append({
                    "chunk_id": chunk_count,
                    "model_size": model_size,
                    "beam_size": beam_size,
                    "sep_latency_ms": round(sep_ms, 1),
                    "asr_latency_ms": round(asr_ms_total, 1),
                    "total_latency_ms": round(total_ms, 1),
                    "chunk_duration_sec": chunk_sec,
                    "n_speakers": sources.shape[0],
                })

                logger.info(
                    f"  Chunk {chunk_count+1}: "
                    f"sep={sep_ms:.0f}ms, asr={asr_ms_total:.0f}ms, "
                    f"total={total_ms:.0f}ms"
                )
                chunk_count += 1

        except Exception as e:
            logger.error(f"Error processing {wav_path.name}: {e}")

    return rows


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_models:
        model_sizes = ["tiny", "base", "small", "medium"]
    else:
        model_sizes = [args.model_size]

    all_rows = []
    for size in model_sizes:
        rows = benchmark_single_model(
            model_size=size,
            data_dir=data_dir,
            n_chunks=args.n_chunks,
            chunk_sec=args.chunk_sec,
            beam_size=args.beam_size,
        )
        all_rows.extend(rows)

    # Save results
    output_csv = output_dir / "latency_benchmark.csv"
    if all_rows:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    import numpy as _np
    if args.compare_models:
        logger.info("=" * 70)
        logger.info("Model Size Comparison (mean / P95 total latency, warm-up excluded):")
        for size in model_sizes:
            size_rows = [r for r in all_rows if r["model_size"] == size]
            if size_rows:
                lat = _np.array([r["total_latency_ms"] for r in size_rows])
                logger.info(
                    f"  whisper-{size:8s}: "
                    f"mean={lat.mean():.0f}ms, "
                    f"p95={_np.percentile(lat, 95):.0f}ms"
                )
    else:
        latencies = [r["total_latency_ms"] for r in all_rows]
        if latencies:
            lat = _np.array(latencies)
            p95 = float(_np.percentile(lat, 95))
            logger.info(f"P95 latency: {p95:.0f}ms (target: <3000ms)")
            if p95 < 3000:
                logger.info("P95 latency target MET (<3s/chunk)")
            else:
                logger.warning(f"P95 latency target MISSED ({p95:.0f}ms > 3000ms)")

    logger.info(f"Results: {output_csv}")


if __name__ == "__main__":
    main()
