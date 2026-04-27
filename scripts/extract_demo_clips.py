#!/usr/bin/env python3
"""Extract short demo clips (30–60 s) from data/dev sessions for the Streamlit demo.

Selects sessions with high speaker overlap (>= 2 speakers, non-silent activity)
and exports a 30–60 second window starting from the most "interesting" region
(highest short-time energy variation, indicating active conversation with overlap).

Usage:
    python scripts/extract_demo_clips.py --n 5
    python scripts/extract_demo_clips.py --n 3 --clip-sec 40 --start-sec 30
    python scripts/extract_demo_clips.py --source data/eval1 --n 5

Output:
    data/demo/clips/  — 4-channel WAV files ready for Streamlit demo
    data/demo/clips/manifest.json  — clip metadata (session, duration, n_speakers)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchaudio


REPO_ROOT = Path(__file__).resolve().parents[1]


def energy_variation(waveform: torch.Tensor, window_sec: float = 1.0, sr: int = 16000) -> float:
    """Return standard deviation of per-second RMS energy (proxy for conversational activity)."""
    win = int(window_sec * sr)
    mono = waveform.mean(dim=0)  # [T]
    n_wins = mono.shape[0] // win
    if n_wins < 2:
        return 0.0
    rms_per_win = torch.tensor([
        (mono[i * win:(i + 1) * win] ** 2).mean().sqrt().item()
        for i in range(n_wins)
    ])
    return float(rms_per_win.std().item())


def find_best_window_start(
    waveform: torch.Tensor,
    clip_samples: int,
    sr: int,
    step_sec: float = 10.0,
) -> int:
    """Find the start sample that maximises energy variation in a clip_samples window."""
    step = int(step_sec * sr)
    T = waveform.shape[-1]
    best_start = 0
    best_var = -1.0
    start = 0
    while start + clip_samples <= T:
        clip = waveform[:, start:start + clip_samples]
        var = energy_variation(clip, sr=sr)
        if var > best_var:
            best_var = var
            best_start = start
        start += step
    return best_start


def extract_clips(
    source_dir: Path,
    dest_dir: Path,
    n_clips: int = 5,
    clip_sec: float = 40.0,
    start_sec: float = 0.0,
    sr: int = 16000,
) -> list[dict]:
    """Extract the most active clips from source WAV files.

    Returns list of clip metadata dicts.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = source_dir / "wav"
    if not wav_dir.is_dir():
        print(f"[error] WAV directory not found: {wav_dir}")
        return []

    wav_files = sorted(wav_dir.glob("*.wav"))
    if not wav_files:
        print(f"[error] No WAV files in {wav_dir}")
        return []

    clip_samples = int(clip_sec * sr)

    # Score each session by energy variation (proxy for overlap activity)
    scored: list[tuple[float, Path]] = []
    for wav in wav_files:
        try:
            wf, file_sr = torchaudio.load(str(wav))
            if file_sr != sr:
                wf = torchaudio.functional.resample(wf, file_sr, sr)
            if wf.shape[-1] < clip_samples + int(start_sec * sr):
                continue  # session too short
            var = energy_variation(wf, sr=sr)
            scored.append((var, wav))
        except Exception as e:
            print(f"[skip] {wav.name}: {e}")

    # Sort by activity (descending) and pick top n_clips
    scored.sort(key=lambda x: -x[0])
    selected = scored[:n_clips]

    manifest: list[dict] = []
    for rank, (var, wav) in enumerate(selected):
        try:
            wf, file_sr = torchaudio.load(str(wav))
            if file_sr != sr:
                wf = torchaudio.functional.resample(wf, file_sr, sr)

            # Find the most active window
            if start_sec > 0.0:
                fixed_start = int(start_sec * sr)
            else:
                fixed_start = find_best_window_start(wf, clip_samples, sr)

            clip = wf[:, fixed_start:fixed_start + clip_samples]
            actual_dur = clip.shape[-1] / sr

            out_name = f"demo_clip_{rank+1:02d}_{wav.stem}.wav"
            out_path = dest_dir / out_name
            torchaudio.save(str(out_path), clip, sr)

            meta = {
                "rank": rank + 1,
                "source": str(wav.relative_to(REPO_ROOT)),
                "clip_file": out_name,
                "clip_path": str(out_path.relative_to(REPO_ROOT)),
                "duration_sec": round(actual_dur, 1),
                "start_sec": round(fixed_start / sr, 1),
                "energy_variation": round(var, 4),
                "n_channels": int(clip.shape[0]),
                "sample_rate": sr,
            }
            manifest.append(meta)
            print(f"  [{rank+1}] {out_name}  ({actual_dur:.0f}s, var={var:.3f})")

        except Exception as e:
            print(f"[error] {wav.name}: {e}")

    manifest_path = dest_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nManifest → {manifest_path}")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Extract demo clips from AISHELL-5 sessions")
    p.add_argument("--source", type=Path, default=REPO_ROOT / "data/dev",
                   help="Source split directory (default: data/dev)")
    p.add_argument("--dest", type=Path, default=REPO_ROOT / "data/demo/clips",
                   help="Output directory (default: data/demo/clips)")
    p.add_argument("--n", type=int, default=5, help="Number of clips to extract")
    p.add_argument("--clip-sec", type=float, default=40.0,
                   help="Clip duration in seconds (default: 40)")
    p.add_argument("--start-sec", type=float, default=0.0,
                   help="Force clip start offset (0 = auto-detect most active window)")
    args = p.parse_args()

    print(f"Extracting {args.n} demo clips from {args.source}")
    clips = extract_clips(
        args.source, args.dest, n_clips=args.n,
        clip_sec=args.clip_sec, start_sec=args.start_sec,
    )
    print(f"\nDone: {len(clips)} clips → {args.dest}")


if __name__ == "__main__":
    main()
