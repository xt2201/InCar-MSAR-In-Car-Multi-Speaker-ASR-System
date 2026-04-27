#!/usr/bin/env python3
"""Materialize AISHELL-5 OpenSLR tree (session/DA*, session/DX01-04C01, …) into
data/{split}/wav/*.wav (4-channel) + data/{split}/text/*.txt (SPK* references).

The public release stores one channel per far-field mic (DX01..DX04C01).
This script stacks them into a single 4-channel WAV per session segment.

Usage:
  python scripts/materialize_aishell5_flat.py --source data/Eval1 --dest data/eval1
  python scripts/materialize_aishell5_flat.py --source data/Dev --dest data/dev

Segment export (per-utterance clips for segment-level evaluation):
  python scripts/materialize_aishell5_flat.py \\
      --source data/Dev --dest data/dev --export-segments
  Produces:
    data/dev/segments/{session_id}/spk{N}_{start_ms}-{end_ms}.wav  (1-channel mono)
    data/dev/segments/{session_id}/refs.json  {spk_id: [{"start":…,"end":…,"text":…}]}
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torchaudio


# ---------------------------------------------------------------------------
# TextGrid parsing
# ---------------------------------------------------------------------------

def parse_textgrid_references(path: Path) -> dict[str, str]:
    """Extract per-speaker joined text from a Praat TextGrid (AISHELL-5)."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    # Item blocks: name = "P0127" ... intervals with text = "..."
    items = re.split(r"\n\s*item \[\d+\]:\s*\n", raw)
    refs: dict[str, str] = {}
    spk_idx = 0
    for block in items[1:]:
        mname = re.search(r'name = "([^"]+)"', block)
        if not mname:
            continue
        texts: list[str] = []
        for tm in re.finditer(r'text = "([^"]*)"', block, re.DOTALL):
            t = tm.group(1).replace("\\n", " ").strip()
            if t:
                texts.append(t)
        if not texts:
            continue
        spk_idx += 1
        key = f"SPK{spk_idx}"
        refs[key] = "".join(texts)  # Chinese: no spaces
    return refs


def parse_textgrid_segments(path: Path) -> dict[str, list[dict]]:
    """Extract per-speaker timed intervals from a Praat TextGrid.

    Returns
    -------
    segments : dict  spk_id -> [{"start": float, "end": float, "text": str}, …]
        Only intervals with non-empty text are included.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    items = re.split(r"\n\s*item \[\d+\]:\s*\n", raw)
    segments: dict[str, list[dict]] = {}
    spk_idx = 0
    for block in items[1:]:
        mname = re.search(r'name = "([^"]+)"', block)
        if not mname:
            continue
        # Find all interval blocks: xmin, xmax, text
        interval_blocks = re.split(r"\n\s*intervals \[\d+\]:\s*\n", block)
        spk_segs: list[dict] = []
        for iv in interval_blocks[1:]:
            xmin_m = re.search(r"xmin\s*=\s*([\d.]+)", iv)
            xmax_m = re.search(r"xmax\s*=\s*([\d.]+)", iv)
            text_m = re.search(r'text\s*=\s*"([^"]*)"', iv, re.DOTALL)
            if not (xmin_m and xmax_m and text_m):
                continue
            text = text_m.group(1).replace("\\n", " ").strip()
            if not text:
                continue
            spk_segs.append({
                "start": float(xmin_m.group(1)),
                "end": float(xmax_m.group(1)),
                "text": text,
            })
        if not spk_segs:
            continue
        spk_idx += 1
        segments[f"SPK{spk_idx}"] = spk_segs
    return segments


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def stack_four_channel(
    session_dir: Path,
    clip: str = "C01",
) -> tuple[torch.Tensor, int]:
    """Load DX01..DX04 *{clip}.wav and return [4, T] and sample rate."""
    paths = [session_dir / f"DX0{i}{clip}.wav" for i in range(1, 5)]
    if not all(p.exists() for p in paths):
        raise FileNotFoundError(f"Missing DX01-04 for {clip} in {session_dir}")
    chunks: list[torch.Tensor] = []
    sr0 = 0
    for p in paths:
        w, sr = torchaudio.load(str(p))
        if w.shape[0] != 1:
            raise ValueError(f"Expected mono far-field file, got {w.shape} at {p}")
        sr0 = sr
        chunks.append(w[0])
    t = min(c.shape[0] for c in chunks)
    stacked = torch.stack([c[:t] for c in chunks], dim=0)
    return stacked, int(sr0)


# ---------------------------------------------------------------------------
# Session-level materialization
# ---------------------------------------------------------------------------

def materialize(source: Path, dest: Path, split_label: str, dry_run: bool) -> int:
    source = source.resolve()
    dest_wav = dest / "wav"
    dest_txt = dest / "text"
    if not dry_run:
        dest_wav.mkdir(parents=True, exist_ok=True)
        dest_txt.mkdir(parents=True, exist_ok=True)

    session_dirs = sorted(
        [p for p in source.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    n_out = 0
    for sess in session_dirs:
        tg = sess / "DX01C01.TextGrid"
        if not tg.exists():
            continue
        try:
            wf, sr = stack_four_channel(sess, "C01")
        except (FileNotFoundError, ValueError) as e:
            print(f"[skip] {sess.name}: {e}")
            continue
        sid = f"{split_label}_{sess.name}_C01"
        out_wav = dest_wav / f"{sid}.wav"
        refs = parse_textgrid_references(tg)
        if not refs:
            print(f"[warn] no text tiers in {tg}, still writing wav")
        if dry_run:
            print(f"would write {out_wav} shape {wf.shape} sr={sr}")
            n_out += 1
            continue
        torchaudio.save(str(out_wav), wf, sr)
        out_txt = dest_txt / f"{sid}.txt"
        lines = [f"{k}: {v}\n" for k, v in sorted(refs.items())]
        out_txt.write_text("".join(lines), encoding="utf-8")
        n_out += 1
        print(f"OK {sid} ch={wf.shape[0]} len_s={wf.shape[1]/sr:.1f}")
    return n_out


# ---------------------------------------------------------------------------
# Segment-level export  (per-utterance WAV clips + JSON refs)
# ---------------------------------------------------------------------------

def export_segments(
    source: Path,
    dest: Path,
    split_label: str,
    driver_channel: int = 0,
    dry_run: bool = False,
    min_duration_sec: float = 0.2,
) -> int:
    """Export per-utterance audio clips aligned from TextGrid intervals.

    For each session that has a TextGrid, writes:
      {dest}/segments/{sid}/spk{N}_{start_ms}-{end_ms}.wav   (mono, channel=driver_channel)
      {dest}/segments/{sid}/refs.json

    The WAV is a single channel (``driver_channel``) of the 4-ch far-field
    recording, trimmed to the utterance interval.  This gives short, clean
    segments (~2–15 s) suitable for segment-level WER evaluation without
    triggering Whisper hallucination.

    Parameters
    ----------
    source : Path
        Raw OpenSLR tree root (e.g. ``data/Dev``).
    dest : Path
        Output root (e.g. ``data/dev``).
    split_label : str
        Prefix for session IDs.
    driver_channel : int
        Which of the 4 far-field channels to extract (0 = driver door mic).
    dry_run : bool
        Print what would be written without writing.
    min_duration_sec : float
        Minimum utterance duration to include.
    """
    source = source.resolve()
    dest_segs = dest / "segments"
    if not dry_run:
        dest_segs.mkdir(parents=True, exist_ok=True)

    session_dirs = sorted(
        [p for p in source.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    n_clips = 0
    for sess in session_dirs:
        tg = sess / "DX01C01.TextGrid"
        if not tg.exists():
            continue
        try:
            wf, sr = stack_four_channel(sess, "C01")
        except (FileNotFoundError, ValueError) as e:
            print(f"[skip segments] {sess.name}: {e}")
            continue

        sid = f"{split_label}_{sess.name}_C01"
        spk_segments = parse_textgrid_segments(tg)
        if not spk_segments:
            print(f"[warn segments] no timed intervals in {tg}")
            continue

        if dry_run:
            total = sum(len(segs) for segs in spk_segments.values())
            print(f"would write {total} clips for {sid}")
            n_clips += total
            continue

        sess_out = dest_segs / sid
        sess_out.mkdir(parents=True, exist_ok=True)

        # Use driver_channel mono track for ASR clips
        ch = min(driver_channel, wf.shape[0] - 1)
        mono = wf[ch]  # [T]

        refs_json: dict[str, list[dict]] = {}
        for spk_id, segs in spk_segments.items():
            spk_num = re.sub(r"[^0-9]", "", spk_id)
            refs_json[spk_id] = []
            for seg in segs:
                dur = seg["end"] - seg["start"]
                if dur < min_duration_sec:
                    continue
                s_idx = max(0, int(seg["start"] * sr))
                e_idx = min(mono.shape[0], int(seg["end"] * sr))
                if e_idx <= s_idx:
                    continue
                clip = mono[s_idx:e_idx].unsqueeze(0)  # [1, T]
                start_ms = int(seg["start"] * 1000)
                end_ms = int(seg["end"] * 1000)
                clip_name = f"spk{spk_num}_{start_ms}-{end_ms}.wav"
                torchaudio.save(str(sess_out / clip_name), clip, sr)
                refs_json[spk_id].append({
                    "clip": clip_name,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                })
                n_clips += 1

        (sess_out / "refs.json").write_text(
            json.dumps(refs_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"OK segments {sid}: {n_clips} clips so far")

    return n_clips


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True, help="e.g. data/Eval1")
    ap.add_argument("--dest", type=Path, required=True, help="e.g. data/eval1")
    ap.add_argument(
        "--label",
        type=str,
        default="",
        help="stem prefix (default: derived from dest name, e.g. eval1)",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--export-segments",
        action="store_true",
        help="Also export per-utterance segment clips for segment-level WER evaluation",
    )
    ap.add_argument(
        "--driver-channel",
        type=int,
        default=0,
        help="Which far-field channel to use for segment clips (default: 0 = driver mic)",
    )
    ap.add_argument(
        "--min-duration",
        type=float,
        default=0.2,
        help="Minimum utterance duration in seconds to include in segment export (default: 0.2)",
    )
    args = ap.parse_args()
    label = args.label or args.dest.name
    n = materialize(args.source, args.dest, label, args.dry_run)
    print(f"Done: {n} sessions materialized → {args.dest}")

    if args.export_segments:
        n_clips = export_segments(
            args.source,
            args.dest,
            label,
            driver_channel=args.driver_channel,
            dry_run=args.dry_run,
            min_duration_sec=args.min_duration,
        )
        print(f"Done: {n_clips} segment clips → {args.dest}/segments/")


if __name__ == "__main__":
    main()
