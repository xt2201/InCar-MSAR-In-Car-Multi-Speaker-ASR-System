#!/usr/bin/env python3
"""Ablation study runner.

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
Runs different pipeline configurations to isolate the contribution
of each component.

Usage:
    python run_ablation.py --study no-separation
    python run_ablation.py --study wav2vec2-asr
    python run_ablation.py --study speaker-cls-comparison
    python run_ablation.py --all
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from loguru import logger

from src.pipeline.data_loader import AISHELL5Loader
from src.evaluation.metrics import compute_wer, compute_cpwer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation study runner")
    p.add_argument("--study", choices=[
        "no-separation", "wav2vec2-asr", "speaker-cls-comparison"
    ], help="Which ablation study to run")
    p.add_argument("--all", action="store_true", help="Run all ablation studies")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--data-dir", default="data/eval1")
    p.add_argument("--output-dir", default="outputs/ablation")
    return p.parse_args()


def ablation_no_separation(samples, output_dir: Path) -> None:
    """Ablation: pipeline without speech separation (direct ASR on mix)."""
    from src.asr.whisper_asr import WhisperASR
    from src.utils.audio import load_multichannel_audio, mix_channels

    logger.info("Ablation: No Separation (direct ASR on mono mix)")
    asr = WhisperASR(model_id="openai/whisper-small", language="zh")
    rows = []

    for i, sample in enumerate(samples):
        try:
            waveform, sr = load_multichannel_audio(sample.wav_path)
            mono = mix_channels(waveform).squeeze()  # [T]

            result = asr.transcribe(mono, sample_rate=sr)
            hyp = result["text"]
            refs = [sample.references[k] for k in sorted(sample.references.keys())] if sample.references else []

            # For single-output vs multi-ref, compute cpWER with 1 hyp vs N refs
            cpwer = compute_cpwer([hyp], refs) if refs else float("nan")
            # Also compute WER against concatenated reference
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


def ablation_wav2vec2(samples, output_dir: Path) -> None:
    """Ablation: replace Whisper with Wav2Vec2 for ASR."""
    logger.info("Ablation: Wav2Vec2 ASR (replaces Whisper)")

    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        import torch
    except ImportError:
        logger.error("transformers not installed for Wav2Vec2")
        return

    from src.separation.separator import SpeechSeparator
    from src.utils.audio import load_multichannel_audio, mix_channels

    model_id = "facebook/wav2vec2-large-xlsr-53-chinese-zh-cn"
    logger.info(f"Loading {model_id}...")

    try:
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load Wav2Vec2: {e}")
        return

    separator = SpeechSeparator(n_speakers=2)
    rows = []

    for i, sample in enumerate(samples):
        try:
            waveform, sr = load_multichannel_audio(sample.wav_path)
            mono = mix_channels(waveform)
            sep_result = separator.separate(mono)
            sources = sep_result["sources"]

            # Transcribe with Wav2Vec2
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


def ablation_speaker_comparison(samples, output_dir: Path) -> None:
    """Ablation: rule-based vs ECAPA speaker classification accuracy."""
    logger.info("Ablation: Speaker Classification Comparison (rule vs ECAPA)")

    from src.separation.separator import SpeechSeparator
    from src.speaker.classifier import RuleBasedRoleClassifier, ECAPAEmbedding, SpeakerRoleClassifier
    from src.utils.audio import load_multichannel_audio, mix_channels

    separator = SpeechSeparator(n_speakers=2)
    rule_cls = RuleBasedRoleClassifier(driver_channel=0)
    ecapa = ECAPAEmbedding()
    hybrid_cls = SpeakerRoleClassifier(ecapa=ecapa, rule_classifier=rule_cls)

    rows = []
    for i, sample in enumerate(samples):
        try:
            waveform, sr = load_multichannel_audio(sample.wav_path)
            mono = mix_channels(waveform)
            sep_result = separator.separate(mono)
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
        "NOTE: Computing accuracy requires ground-truth speaker role annotations. "
        "AISHELL-5 does not provide explicit Driver/Passenger labels per session."
    )


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        logger.warning(f"No rows to write to {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = AISHELL5Loader(args.data_dir, max_samples=args.n)
    samples = list(loader)
    logger.info(f"Loaded {len(samples)} samples for ablation study")

    studies = []
    if args.all:
        studies = ["no-separation", "wav2vec2-asr", "speaker-cls-comparison"]
    elif args.study:
        studies = [args.study]
    else:
        logger.error("Specify --study or --all")
        raise SystemExit(1)

    for study in studies:
        if study == "no-separation":
            ablation_no_separation(samples, output_dir)
        elif study == "wav2vec2-asr":
            ablation_wav2vec2(samples, output_dir)
        elif study == "speaker-cls-comparison":
            ablation_speaker_comparison(samples, output_dir)


if __name__ == "__main__":
    main()
