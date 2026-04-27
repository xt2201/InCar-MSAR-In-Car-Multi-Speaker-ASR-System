"""AISHELL-5 dataset loader and transcript parser.

This loader expects a **materialized** split (see `docs/dataset.md` §9, `materialize_aishell5_flat.py`):
``{data_dir}/wav/*.wav`` (4-channel, 16 kHz) and ``{data_dir}/text/{session_id}.txt`` with lines
``SPK1: ...``, ``SPK2: ...``.

Raw OpenSLR session trees (``DX01C01.wav``… per folder) are not read here — run materialization first.

Splits *dev, eval1, eval2* are for WER/cpWER evaluation. The **noise** split (environmental recording only) has
no word transcripts and is not loaded by this class for the standard benchmark.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger


@dataclass
class Sample:
    """A single recording sample from AISHELL-5.

    Attributes
    ----------
    session_id : str
        Unique session/recording identifier.
    wav_path : Path
        Path to 4-channel WAV file.
    transcript_path : Path or None
        Path to transcript text file.
    references : dict[str, str]
        Mapping from speaker ID to reference transcript.
        e.g. {"SPK1": "我们现在去哪里？", "SPK2": "去公司吧"}
    n_speakers : int
        Number of speakers in this recording.
    """
    session_id: str
    wav_path: Path
    transcript_path: Optional[Path] = None
    references: dict[str, str] = field(default_factory=dict)
    n_speakers: int = 2


class AISHELL5Loader:
    """Load and iterate over AISHELL-5 dataset samples.

    Parameters
    ----------
    data_dir : str or Path
        Root directory for a split, e.g. "data/dev".
        Expected structure:
          {data_dir}/wav/*.wav
          {data_dir}/text/*.txt
    max_samples : int or None
        Limit number of samples loaded (None = all).
    """

    def __init__(
        self,
        data_dir: str | Path,
        max_samples: Optional[int] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples

        self.wav_dir = self.data_dir / "wav"
        self.text_dir = self.data_dir / "text"

        if not self.wav_dir.exists():
            raise FileNotFoundError(f"WAV directory not found: {self.wav_dir}")

        self._samples: list[Sample] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Scan directory and build sample list."""
        wav_files = sorted(self.wav_dir.glob("*.wav"))

        if self.max_samples is not None:
            wav_files = wav_files[:self.max_samples]

        for wav_path in wav_files:
            session_id = wav_path.stem
            txt_path = self.text_dir / f"{session_id}.txt"

            references = {}
            if txt_path.exists():
                references = self._parse_transcript(txt_path)

            sample = Sample(
                session_id=session_id,
                wav_path=wav_path,
                transcript_path=txt_path if txt_path.exists() else None,
                references=references,
                n_speakers=len(references) if references else 2,
            )
            self._samples.append(sample)

        logger.info(
            f"AISHELL5Loader: {len(self._samples)} samples loaded from {self.data_dir}"
        )

    def _parse_transcript(self, txt_path: Path) -> dict[str, str]:
        """Parse AISHELL-5 transcript file.

        Expected format:
            SPK1: 我们现在去哪里？
            SPK2: 去公司吧
            SPK1: 好的

        Returns
        -------
        references : dict mapping speaker_id -> concatenated transcript.
        """
        speaker_texts: dict[str, list[str]] = {}

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Match "SPK1: text" or "S1: text" or "SPEAKER_1: text"
                match = re.match(r"^(SPK\d+|S\d+|SPEAKER_\d+)[:\s]+(.+)$", line, re.IGNORECASE)
                if match:
                    spk_id = match.group(1).upper()
                    text = match.group(2).strip()
                    if spk_id not in speaker_texts:
                        speaker_texts[spk_id] = []
                    speaker_texts[spk_id].append(text)

        # Concatenate all utterances per speaker
        return {spk: " ".join(texts) for spk, texts in speaker_texts.items()}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Sample:
        return self._samples[idx]

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    def statistics(self) -> dict:
        """Return dataset statistics."""
        n_speakers_dist: dict[int, int] = {}
        for s in self._samples:
            n = s.n_speakers
            n_speakers_dist[n] = n_speakers_dist.get(n, 0) + 1

        return {
            "total_samples": len(self._samples),
            "n_speakers_distribution": n_speakers_dist,
            "has_transcripts": sum(1 for s in self._samples if s.references),
            "data_dir": str(self.data_dir),
        }
