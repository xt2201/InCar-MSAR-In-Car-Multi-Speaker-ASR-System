#!/usr/bin/env python3
"""Materialize AISHELL-5 OpenSLR tree (session/DA*, session/DX01-04C01, …) into
data/{split}/wav/*.wav (4-channel) + data/{split}/text/*.txt (SPK* references).

The public release stores one channel per far-field mic (DX01C01 … DX04C01).
This script stacks them into a single 4-channel WAV per session segment.

Usage:
  python scripts/materialize_aishell5_flat.py --source data/Eval1 --dest data/eval1
  python scripts/materialize_aishell5_flat.py --source data/Dev --dest data/dev
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
import torchaudio


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
    args = ap.parse_args()
    label = args.label or args.dest.name
    n = materialize(args.source, args.dest, label, args.dry_run)
    print(f"Done: {n} sessions materialized → {args.dest}")


if __name__ == "__main__":
    main()
