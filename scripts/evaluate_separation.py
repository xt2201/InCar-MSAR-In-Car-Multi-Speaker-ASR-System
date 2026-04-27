#!/usr/bin/env python3
"""Evaluate speech separation quality (proxy metrics) on AISHELL-5 dev set.

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
On **CPU**, uses `ChannelSelectSeparator` (instant, no GPU).
On **GPU**, uses SepFormer when `method=sepformer`.

Usage:
    python evaluate_separation.py --n 20
    python evaluate_separation.py --n 20 --method sepformer   # GPU recommended
    python evaluate_separation.py --n 20 --method channel_select
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from loguru import logger

from src.pipeline.data_loader import AISHELL5Loader
from src.separation import get_separator, SpeechSeparator
from src.utils.audio import load_multichannel_audio, mix_channels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate separation (proxy metrics)")
    p.add_argument("--n", type=int, default=20, help="Number of samples")
    p.add_argument(
        "--method",
        default="auto",
        choices=["auto", "channel_select", "ica", "sepformer", "convtasnet"],
        help="Separation backend. 'auto' = SepFormer on GPU, channel_select on CPU.",
    )
    p.add_argument("--compare", action="store_true",
                   help="Compare channel_select vs sepformer (GPU only for latter)")
    p.add_argument("--data-dir", default="data/dev")
    p.add_argument("--output-dir", default="outputs/metrics")
    return p.parse_args()


def evaluate_one(
    method: str,
    samples,
    output_dir: Path,
) -> list[dict]:
    has_gpu = torch.cuda.is_available()
    if method == "sepformer" and not has_gpu:
        logger.warning("SepFormer on CPU is not supported in this script; using channel_select.")
        method = "channel_select"

    if method == "convtasnet":
        separator = SpeechSeparator(model_name="convtasnet", n_speakers=2, device="auto")
    elif method in ("auto", "channel_select", "ica"):
        separator = get_separator(method=method, n_speakers=2)
    else:
        separator = get_separator(method="sepformer", n_speakers=2, model_name="sepformer-wsj02mix")

    rows = []

    for i, sample in enumerate(samples):
        try:
            waveform, sr = load_multichannel_audio(sample.wav_path)

            if isinstance(separator, SpeechSeparator):
                mono = mix_channels(waveform)
                result = separator.separate(mono)
            else:
                # CPU separators: full 4-channel mix
                result = separator.separate(waveform)

            sources = result["sources"]

            energies = (sources ** 2).mean(dim=-1)
            energy_ratio = (energies.max() / (energies.min() + 1e-8)).item()

            rows.append({
                "file_id": sample.session_id,
                "method": method,
                "n_speakers": sources.shape[0],
                "energy_ratio_proxy": round(energy_ratio, 3),
                "inference_ms": round(result["inference_time_ms"], 1),
                "si_snri": "N/A (near-field ref only in train split)",
            })

            logger.info(
                f"[{i+1}/{len(samples)}] {sample.session_id}: "
                f"energy_ratio={energy_ratio:.2f}, t={result['inference_time_ms']:.0f}ms"
            )

        except Exception as e:
            logger.error(f"Error on {sample.session_id}: {e}")
            rows.append({
                "file_id": sample.session_id,
                "method": method,
                "error": str(e),
            })

    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = AISHELL5Loader(args.data_dir, max_samples=args.n)
    samples = list(loader)

    if args.compare:
        methods = ["channel_select", "sepformer"] if torch.cuda.is_available() else ["channel_select"]
    else:
        methods = [args.method]

    all_rows = []
    for method in methods:
        logger.info(f"Evaluating method: {method}")
        rows = evaluate_one(method, samples, output_dir)
        all_rows.extend(rows)

    output_csv = output_dir / "separation_metrics.csv"
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

    logger.info(f"Separation evaluation saved to {output_csv}")


if __name__ == "__main__":
    main()
