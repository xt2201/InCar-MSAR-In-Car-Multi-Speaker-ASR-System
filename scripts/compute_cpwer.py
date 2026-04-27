#!/usr/bin/env python3
"""Compute cpWER (Concatenated Permutation WER) for multi-speaker ASR.

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
This script loads hypothesis and reference transcripts and computes
cpWER using the Hungarian algorithm for optimal speaker matching.

Usage:
    python compute_cpwer.py --hyp outputs/metrics/wer_pipeline_eval1.csv \
                            --ref-dir data/eval1/text/
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from loguru import logger

from src.evaluation.metrics import compute_wer, compute_cpwer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute cpWER for multi-speaker ASR")
    p.add_argument("--hyp", help="CSV with hypothesis transcripts (file_id, speaker_id, hypothesis)")
    p.add_argument("--ref-dir", help="Directory containing reference text files")
    p.add_argument("--output", default="outputs/metrics/cpwer_results.csv")
    return p.parse_args()


def load_hypotheses_from_csv(csv_path: str) -> dict[str, list[str]]:
    """Load hypotheses from CSV. Returns dict: session_id -> list of hyp strings."""
    sessions = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("hypothesis"):
                sessions[row["file_id"]].append(row["hypothesis"])
    return dict(sessions)


def load_references_from_dir(ref_dir: str) -> dict[str, list[str]]:
    """Load references from text files in directory."""
    from src.pipeline.data_loader import AISHELL5Loader
    # Construct loader pointing to text subdir
    ref_path = Path(ref_dir)
    sessions = {}

    for txt_file in sorted(ref_path.glob("*.txt")):
        session_id = txt_file.stem
        speaker_texts: dict[str, list[str]] = {}

        import re
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                match = re.match(r"^(SPK\d+|S\d+)[:\s]+(.+)$", line, re.IGNORECASE)
                if match:
                    spk = match.group(1).upper()
                    text = match.group(2).strip()
                    if spk not in speaker_texts:
                        speaker_texts[spk] = []
                    speaker_texts[spk].append(text)

        sessions[session_id] = [
            " ".join(texts) for k, texts in sorted(speaker_texts.items())
        ]

    return sessions


def main() -> None:
    args = parse_args()

    if not args.hyp:
        logger.error("Must provide --hyp CSV path.")
        raise SystemExit(1)

    if not Path(args.hyp).exists():
        logger.error(f"Hypothesis file not found: {args.hyp}")
        raise SystemExit(1)

    hyp_sessions = load_hypotheses_from_csv(args.hyp)

    ref_sessions = {}
    if args.ref_dir and Path(args.ref_dir).exists():
        ref_sessions = load_references_from_dir(args.ref_dir)
    else:
        logger.warning("No reference directory provided. cpWER cannot be computed without references.")
        raise SystemExit(1)

    results = []
    cpwer_list = []

    for session_id, hyps in hyp_sessions.items():
        refs = ref_sessions.get(session_id, [])
        if not refs:
            logger.warning(f"No reference for session: {session_id}")
            continue

        cpwer = compute_cpwer(hyps, refs)
        cpwer_list.append(cpwer)

        results.append({
            "session_id": session_id,
            "n_hyp_speakers": len(hyps),
            "n_ref_speakers": len(refs),
            "cpwer": round(cpwer, 4),
        })

        logger.info(f"{session_id}: cpWER={cpwer:.1%}")

    # Write results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["session_id", "n_hyp_speakers", "n_ref_speakers", "cpwer"])
        writer.writeheader()
        writer.writerows(results)

    overall_cpwer = np.mean(cpwer_list) if cpwer_list else float("nan")
    logger.info("=" * 50)
    logger.info(f"Overall cpWER: {overall_cpwer:.1%} (n={len(cpwer_list)})")
    logger.info(f"Results saved: {output_path}")


if __name__ == "__main__":
    main()
