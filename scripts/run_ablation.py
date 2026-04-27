#!/usr/bin/env python3
"""Ablation study runner for InCar-MSAR.

Runs different separation backend / ASR model configurations to isolate
each component's contribution to the overall CER/cpWER.

Usage:
    # Separation backend comparison (main ablation table)
    python scripts/run_ablation.py --study sep-backends --n 10

    # No-separation baseline (direct ASR on mono mix)
    python scripts/run_ablation.py --study no-separation --n 10

    # Wav2Vec2 vs Whisper ASR
    python scripts/run_ablation.py --study wav2vec2-asr --n 10

    # Speaker classifier comparison
    python scripts/run_ablation.py --study speaker-cls-comparison --n 10

    # Run all (sequential)
    python scripts/run_ablation.py --all --n 10

Results are saved to outputs/ablation/ as individual CSVs.
Run scripts/generate_tables.py afterwards to produce the LaTeX ablation table.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.data_loader import AISHELL5Loader
from src.evaluation.metrics import compute_wer, compute_cpwer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation study runner")
    p.add_argument("--study", choices=[
        "sep-backends", "no-separation", "wav2vec2-asr", "speaker-cls-comparison"
    ], help="Which ablation study to run")
    p.add_argument("--all", action="store_true", help="Run all ablation studies")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--split", default="eval1")
    p.add_argument("--data-dir", default="data/eval1")
    p.add_argument("--output-dir", default="outputs/ablation")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Ablation 1: Separation backend comparison (the core ablation table)
# ---------------------------------------------------------------------------

def ablation_sep_backends(n: int, split: str, output_dir: Path) -> None:
    """Compare baseline / channel_select / beamform / sepformer / upper_bound.

    Calls evaluate.py with --sep-method for each backend so that results are
    stored in outputs/metrics/ using the standard naming convention.
    Results are also summarised here and saved to outputs/ablation/.
    """
    logger.info("Ablation: Separation backend comparison")

    configs = [
        # (label, evaluate.py mode, sep-method arg)
        ("baseline",      "baseline",    "auto"),
        ("channel_select", "pipeline",   "channel_select"),
        ("beamform",       "pipeline",   "beamform"),
        ("sepformer",      "pipeline",   "sepformer"),
        ("upper_bound",   "upper_bound", "auto"),
    ]

    summary_rows = []
    for label, mode, sep_method in configs:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "evaluate.py"),
            "--split", split,
            "--mode", mode,
            "--n", str(n),
            "--config", str(REPO_ROOT / "configs/default.yaml"),
            "--output-dir", str(REPO_ROOT / "outputs/metrics"),
        ]
        if sep_method != "auto" and mode == "pipeline":
            cmd += ["--sep-method", sep_method]

        logger.info(f"Running: {' '.join(cmd[-6:])}")
        try:
            result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=False, text=True)
            if result.returncode != 0:
                logger.warning(f"evaluate.py returned {result.returncode} for {label}")
        except Exception as e:
            logger.error(f"Failed to run ablation for {label}: {e}")
            continue

        # Load summary JSON
        sep_suffix = f"_{sep_method}" if sep_method != "auto" else ""
        summary_path = REPO_ROOT / "outputs/metrics" / f"summary_{mode}{sep_suffix}_{split}.json"
        if summary_path.exists():
            with open(summary_path) as f:
                s = json.load(f)
            summary_rows.append({
                "label": label,
                "mode": mode,
                "sep_method": sep_method,
                "wer_mean": round(s.get("wer", {}).get("mean", float("nan")), 4),
                "cpwer_mean": round(s.get("cpwer", {}).get("mean", float("nan")), 4),
                "latency_mean_ms": s.get("latency", {}).get("mean_ms", 0),
                "n_samples": s.get("n_samples", 0),
            })
        else:
            logger.warning(f"Summary not found: {summary_path}")
            summary_rows.append({
                "label": label, "mode": mode, "sep_method": sep_method,
                "wer_mean": float("nan"), "cpwer_mean": float("nan"),
                "latency_mean_ms": 0, "n_samples": 0,
            })

    out_csv = output_dir / "sep_backends.csv"
    _write_csv(summary_rows, out_csv)
    logger.info(f"Separation ablation summary → {out_csv}")
    for row in summary_rows:
        logger.info(
            f"  {row['label']:20s}  WER={row['wer_mean']:.1%}  cpWER={row['cpwer_mean']:.1%}"
        )


# ---------------------------------------------------------------------------
# Ablation 2: No separation (direct ASR on mono mix)
# ---------------------------------------------------------------------------

def ablation_no_separation(samples, output_dir: Path) -> None:
    """Ablation: pipeline without speech separation (direct ASR on mix)."""
    from src.asr.whisper_asr import WhisperASR
    from src.utils.audio import load_multichannel_audio, mix_channels

    logger.info("Ablation: No Separation (direct ASR on mono mix)")
    asr = WhisperASR(model_id="openai/whisper-small", language="zh", device="auto", beam_size=2)
    rows = []

    for i, sample in enumerate(samples):
        try:
            waveform, sr = load_multichannel_audio(sample.wav_path)
            mono = mix_channels(waveform).squeeze()  # [T]
            result = asr.transcribe(mono, sample_rate=sr)
            hyp = result["text"]
            refs = [sample.references[k] for k in sorted(sample.references.keys())] if sample.references else []
            cpwer = compute_cpwer([hyp], refs) if refs else float("nan")
            concat_ref = " ".join(refs)
            wer = compute_wer(hyp, concat_ref) if concat_ref else float("nan")
            rows.append({
                "file_id": sample.session_id,
                "hypothesis": hyp,
                "references": " | ".join(refs),
                "wer": round(wer, 4),
                "cpwer": round(cpwer, 4),
                "ablation": "no_separation",
            })
            logger.info(f"[{i+1}/{len(samples)}] WER={wer:.1%}, cpWER={cpwer:.1%}")
        except Exception as e:
            logger.error(f"Error: {e}")

    out_csv = output_dir / "no_separation.csv"
    _write_csv(rows, out_csv)
    valid = [r["cpwer"] for r in rows if isinstance(r["cpwer"], float) and r["cpwer"] == r["cpwer"]]
    mean_cpwer = sum(valid) / max(1, len(valid))
    logger.info(f"Ablation no-separation: mean cpWER={mean_cpwer:.1%} → saved {out_csv}")


# ---------------------------------------------------------------------------
# Ablation 3: Wav2Vec2 vs Whisper ASR
# ---------------------------------------------------------------------------

def ablation_wav2vec2(samples, output_dir: Path) -> None:
    """Ablation: replace Whisper with Wav2Vec2-XLSR for ASR."""
    logger.info("Ablation: Wav2Vec2 ASR (replaces Whisper)")

    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        import torch
    except ImportError:
        logger.error("transformers not installed for Wav2Vec2")
        return

    from src.separation.cpu_separator import ChannelSelectSeparator
    from src.utils.audio import load_multichannel_audio

    model_id = "facebook/wav2vec2-large-xlsr-53-chinese-zh-cn"
    logger.info(f"Loading {model_id}...")
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
    except Exception as e:
        logger.error(f"Failed to load Wav2Vec2: {e}")
        return

    separator = ChannelSelectSeparator(n_speakers=2)
    rows = []

    for i, sample in enumerate(samples):
        try:
            waveform, sr = load_multichannel_audio(sample.wav_path)
            sep_result = separator.separate(waveform)
            sources = sep_result["sources"]
            hyps = []
            for src in sources:
                audio_np = src.cpu().numpy()
                inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt", padding=True)
                with torch.no_grad():
                    logits = model(inputs.input_values.to(device)).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcript = processor.batch_decode(predicted_ids)[0]
                hyps.append(transcript)
            refs = list(sample.references.values())
            cpwer = compute_cpwer(hyps, refs) if refs else float("nan")
            rows.append({
                "file_id": sample.session_id,
                "hypotheses": " | ".join(hyps),
                "references": " | ".join(refs),
                "cpwer": round(cpwer, 4),
                "ablation": "wav2vec2_asr",
            })
            logger.info(f"[{i+1}/{len(samples)}] cpWER={cpwer:.1%}")
        except Exception as e:
            logger.error(f"Error: {e}")

    out_csv = output_dir / "wav2vec2_asr.csv"
    _write_csv(rows, out_csv)
    logger.info(f"Ablation wav2vec2: saved {out_csv}")


# ---------------------------------------------------------------------------
# Ablation 4: Speaker classifier comparison
# ---------------------------------------------------------------------------

def ablation_speaker_comparison(samples, output_dir: Path) -> None:
    """Ablation: rule-based vs ECAPA speaker classification accuracy."""
    logger.info("Ablation: Speaker Classification Comparison (rule vs ECAPA)")

    from src.separation.cpu_separator import ChannelSelectSeparator
    from src.speaker.classifier import RuleBasedRoleClassifier, ECAPAEmbedding, SpeakerRoleClassifier
    from src.utils.audio import load_multichannel_audio

    separator = ChannelSelectSeparator(n_speakers=2)
    rule_cls = RuleBasedRoleClassifier(driver_channel=0)
    ecapa = ECAPAEmbedding()
    hybrid_cls = SpeakerRoleClassifier(ecapa=ecapa, rule_classifier=rule_cls)

    rows = []
    for i, sample in enumerate(samples):
        try:
            waveform, sr = load_multichannel_audio(sample.wav_path)
            sep_result = separator.separate(waveform)
            sources = sep_result["sources"]
            rule_roles = rule_cls.classify(sources, waveform)
            hybrid_roles = hybrid_cls.classify(sources, waveform)
            rows.append({
                "file_id": sample.session_id,
                "rule_roles": str(rule_roles),
                "hybrid_roles": str(hybrid_roles),
            })
            logger.info(f"[{i+1}/{len(samples)}] rule={rule_roles}, hybrid={hybrid_roles}")
        except Exception as e:
            logger.error(f"Error: {e}")

    out_csv = output_dir / "speaker_cls_comparison.csv"
    _write_csv(rows, out_csv)
    logger.info(
        f"Speaker comparison done: {out_csv}\n"
        "NOTE: accuracy requires ground-truth labels not in public AISHELL-5 splits."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        logger.warning(f"No rows to write to {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    studies: list[str] = []
    if args.all:
        studies = ["sep-backends", "no-separation", "wav2vec2-asr", "speaker-cls-comparison"]
    elif args.study:
        studies = [args.study]
    else:
        logger.error("Specify --study or --all")
        raise SystemExit(1)

    for study in studies:
        if study == "sep-backends":
            ablation_sep_backends(args.n, args.split, output_dir)
        else:
            loader = AISHELL5Loader(args.data_dir, max_samples=args.n)
            samples = list(loader)
            logger.info(f"Loaded {len(samples)} samples for ablation '{study}'")
            if study == "no-separation":
                ablation_no_separation(samples, output_dir)
            elif study == "wav2vec2-asr":
                ablation_wav2vec2(samples, output_dir)
            elif study == "speaker-cls-comparison":
                ablation_speaker_comparison(samples, output_dir)


if __name__ == "__main__":
    main()
