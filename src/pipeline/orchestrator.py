"""Main pipeline orchestrator for In-Car Multi-Speaker ASR."""
from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
from loguru import logger

from src.separation import SpeechSeparator, ChunkedSeparator, get_separator
from src.separation.cpu_separator import ChannelSelectSeparator, ICASeparator
from src.asr import WhisperASR, ChunkedASR
from src.speaker import RuleBasedRoleClassifier, ECAPAEmbedding, SpeakerRoleClassifier
from src.intent import IntentEngine
from src.utils.audio import (
    load_multichannel_audio,
    mix_channels,
    chunk_audio,
)
from src.utils.config import load_config


class InCarASRPipeline:
    """End-to-end in-car multi-speaker ASR pipeline.

    Pipeline stages (aligns with ``docs/incar_msar_architecture.md``):
    1. Load 4-channel far-field audio (16 kHz) via ``load_multichannel_audio()``.
    2. Chunked speech separation: SepFormer on mono mixture (GPU) *or* CPU
       fallback (``channel_select`` / ``ica``) on full multichannel input.
    3. Optional fallback: if separation quality is too low, use per-mic
       channels as pseudo-sources.
    4. Speaker role classification (rule-based or ECAPA hybrid) on separated
       tracks + original multichannel (driver-channel correlation).
    5. ASR (Whisper) on each separated track, then intent parsing.

    *Intent is the final stage per output row; classifiers use separated audio
    from stage 2 before ASR, matching the “separation → diarization/role → ASR”
    dependency chain used for evaluation and demos.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML config file. Defaults to configs/default.yaml.

    Examples
    --------
    >>> pipeline = InCarASRPipeline("configs/default.yaml")
    >>> results = pipeline.process_file("data/dev/wav/session_001.wav")
    >>> for utterance in results:
    ...     print(f"[{utterance['role']}]: {utterance['transcript']} → {utterance['intent']['intent']}")
    """

    def __init__(
        self,
        config_path: str | Path = "configs/default.yaml",
    ) -> None:
        self.cfg = load_config(config_path)
        self._chunked_sep = None
        self._sep_method = None
        self._asr = None
        self._chunked_asr = None
        self._rule_classifier = None
        self._ecapa = None
        self._speaker_classifier = None
        self._intent_engine = None
        self._active_n_speakers: Optional[int] = None
        self._built_n_speakers: Optional[int] = None
        # Resolved device for ASR/ECAPA (same logical device as Whisper; separation may differ)
        ad = getattr(self.cfg.asr, "device", "auto")
        if ad is None or str(ad) == "auto":
            self._asr_device: str = "cuda" if torch.cuda.is_available() else "cpu"
        elif str(ad) == "cpu":
            self._asr_device = "cpu"
        else:
            # Match WhisperASR: "cuda" without GPU -> cpu
            self._asr_device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"InCarASRPipeline initialized with config: {config_path}")

    # ------------------------------------------------------------------
    # Lazy loading of heavy models
    # ------------------------------------------------------------------

    def _effective_n_speakers(self) -> int:
        """Clamped [2, max_speakers] from runtime override or config."""
        sep_cfg = self.cfg.separation
        max_spk = int(getattr(sep_cfg, "max_speakers", 4))
        if self._active_n_speakers is not None:
            n = int(self._active_n_speakers)
        else:
            n = int(sep_cfg.n_speakers)
        return int(max(2, min(n, max_spk)))

    @property
    def separator(self) -> ChunkedSeparator:
        n_spk = self._effective_n_speakers()
        if self._chunked_sep is not None and self._built_n_speakers is not None and n_spk != self._built_n_speakers:
            self._chunked_sep = None
        if self._chunked_sep is None:
            sep_cfg = self.cfg.separation
            device = sep_cfg.device  # "auto", "cuda", "cpu"
            has_gpu = torch.cuda.is_available()

            cpu_method = getattr(sep_cfg, "cpu_fallback_method", "channel_select")
            if device == "auto":
                self._sep_method = "sepformer" if has_gpu else cpu_method
            else:
                self._sep_method = "sepformer" if (device == "cuda" and has_gpu) else cpu_method

            sep = get_separator(
                method=self._sep_method,
                n_speakers=n_spk,
                model_name=sep_cfg.model,
                model_hub=sep_cfg.model_hub,
                device="auto",
            )
            self._built_n_speakers = n_spk
            logger.info(
                f"Separation method: {self._sep_method} (n_speakers={n_spk}, GPU={'yes' if has_gpu else 'no'})"
            )

            class _WrappedSep:
                def __init__(self, inner):
                    self._inner = inner
                def separate(self, x):
                    return self._inner.separate(x)

            wrapped = _WrappedSep(sep)

            self._chunked_sep = ChunkedSeparator(
                separator=wrapped,
                chunk_sec=self.cfg.audio.chunk_size_sec,
                overlap_sec=self.cfg.audio.chunk_overlap_sec,
                sample_rate=self.cfg.audio.sample_rate,
            )
        return self._chunked_sep

    @property
    def _needs_mono(self) -> bool:
        """SepFormer needs mono input; CPU separators (channel_select/ica/beamform) need multichannel."""
        return getattr(self, "_sep_method", "sepformer") == "sepformer"

    @property
    def asr(self) -> WhisperASR:
        if self._asr is None:
            asr_cfg = self.cfg.asr
            self._asr = WhisperASR(
                model_id=asr_cfg.model,
                language=asr_cfg.language,
                device=asr_cfg.device,
                beam_size=asr_cfg.beam_size,
                batch_size=asr_cfg.batch_size,
                max_new_tokens=asr_cfg.max_new_tokens,
                no_speech_threshold=float(getattr(asr_cfg, "no_speech_threshold", 0.6)),
                compression_ratio_threshold=float(
                    getattr(asr_cfg, "compression_ratio_threshold", 2.4)
                ),
                condition_on_prev_tokens=bool(
                    getattr(asr_cfg, "condition_on_prev_tokens", False)
                ),
            )
        return self._asr

    @property
    def speaker_classifier(self) -> SpeakerRoleClassifier:
        if self._speaker_classifier is None:
            spk_cfg = self.cfg.speaker
            rule = RuleBasedRoleClassifier(
                driver_channel=spk_cfg.driver_channel,
                channel_seat_map=dict(spk_cfg.channel_seat_map),
            )
            # Match architecture doc: ECAPA on same resolved device as Whisper ASR
            ecapa = ECAPAEmbedding(
                model_hub=spk_cfg.ecapa_model,
                device=self._asr_device,
            )
            self._speaker_classifier = SpeakerRoleClassifier(
                ecapa=ecapa,
                rule_classifier=rule,
                cosine_threshold=spk_cfg.cosine_threshold,
            )
        return self._speaker_classifier

    @property
    def intent_engine(self) -> IntentEngine:
        if self._intent_engine is None:
            self._intent_engine = IntentEngine(
                keyword_config=self.cfg.intent.keyword_config
            )
        return self._intent_engine

    # ------------------------------------------------------------------
    # Core processing methods
    # ------------------------------------------------------------------

    def process_file(
        self,
        wav_path: str | Path,
        n_speakers: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Process a single audio file end-to-end.

        Parameters
        ----------
        wav_path : str or Path
            Path to 4-channel WAV file.
        n_speakers : int, optional
            If set, overrides ``separation.n_speakers`` for this file (e.g. from
            the number of reference speakers in AISHELL-5, 2–4). The separator
            is rebuilt if this differs from the previous call.

        Returns
        -------
        utterances : list of dicts, each with:
            {
                "file_id": str,
                "timestamp": float,
                "speaker_id": int,
                "role": str,
                "transcript": str,
                "intent": dict,
                "separation_ms": float,
                "asr_ms": float,
                "total_ms": float,
            }

        Raises
        ------
        FileNotFoundError : If wav_path does not exist.
        ValueError : If audio is too short (< min_duration_sec).
        """
        self._active_n_speakers = n_speakers
        try:
            return self._process_file_impl(wav_path)
        finally:
            self._active_n_speakers = None

    def _process_file_impl(self, wav_path: Path) -> list[dict[str, Any]]:
        wav_path = Path(wav_path)
        file_id = wav_path.stem

        logger.info(f"Processing: {wav_path.name}")
        t_start = time.perf_counter()

        # 1) Load multi-channel (far-field) audio
        waveform, sr = load_multichannel_audio(
            wav_path,
            target_sr=self.cfg.audio.sample_rate,
        )
        logger.debug(f"  Loaded: shape={waveform.shape}, sr={sr}")

        # 2) Separated sources (ChunkedSeparator; mix-to-mono only for SepFormer)
        t_sep = time.perf_counter()
        sep_result = self.separator.process_file(waveform, mix_to_mono=self._needs_mono)
        sep_ms = (time.perf_counter() - t_sep) * 1000
        sources = sep_result["sources"]  # [N, T]

        # 2b) Optional fallback: low separation quality (GPU path / SI-SNRi heuristic)
        if self._should_fallback_separation(waveform, sources):
            logger.warning(
                f"[{file_id}] Low separation quality. Falling back to per-mic sources."
            )
            sources = self._direct_multichannel_asr_sources(waveform)

        # 3) Speaker role classification (on separated tracks + full mixture ref.)
        role_map = self._classify_speakers(sources, waveform)

        # 4) ASR on each track, then intent
        utterances = []
        for spk_id in range(sources.shape[0]):
            t_asr = time.perf_counter()
            asr_result = self.asr.transcribe(
                sources[spk_id],
                sample_rate=sr,
            )
            asr_ms = (time.perf_counter() - t_asr) * 1000

            role = role_map.get(spk_id, f"Passenger_{spk_id}")
            transcript = asr_result["text"]

            # 5. Intent parsing
            intent = self.intent_engine.parse(transcript, speaker=role)

            total_ms = (time.perf_counter() - t_start) * 1000

            utterances.append({
                "file_id": file_id,
                "timestamp": 0.0,
                "speaker_id": spk_id,
                "role": role,
                "transcript": transcript,
                "intent": intent,
                "separation_ms": sep_ms,
                "asr_ms": asr_ms,
                "total_ms": total_ms,
                "is_silent": asr_result.get("is_silent", False),
            })

        logger.info(
            f"  Done: {len(utterances)} speakers, "
            f"sep={sep_ms:.0f}ms, "
            f"total={(time.perf_counter() - t_start) * 1000:.0f}ms"
        )
        return utterances

    def process_chunk(
        self,
        chunk: torch.Tensor,
        chunk_idx: int = 0,
    ) -> list[dict[str, Any]]:
        """Process a single audio chunk (for streaming mode).

        Parameters
        ----------
        chunk : torch.Tensor
            Shape [C, T] where T = chunk_size_samples.
        chunk_idx : int
            Chunk index for timestamp calculation.

        Returns
        -------
        utterances : list of dicts (same schema as process_file).
        """
        t_start = time.perf_counter()
        timestamp = chunk_idx * self.cfg.audio.chunk_size_sec

        # Separate (mono for SepFormer, multichannel for CPU separators)
        t_sep = time.perf_counter()
        if self._needs_mono:
            sep_input = mix_channels(chunk)  # [1, T]
        else:
            sep_input = chunk  # [C, T]
        sep_result = self.separator.separator.separate(sep_input)
        sep_ms = (time.perf_counter() - t_sep) * 1000
        sources = sep_result["sources"]  # [N, T]

        # Speaker role classification
        role_map = self._classify_speakers(sources, chunk)

        utterances = []
        for spk_id in range(sources.shape[0]):
            t_asr = time.perf_counter()
            asr_result = self.asr.transcribe(
                sources[spk_id],
                sample_rate=self.cfg.audio.sample_rate,
            )
            asr_ms = (time.perf_counter() - t_asr) * 1000

            role = role_map.get(spk_id, f"Passenger_{spk_id}")
            transcript = asr_result["text"]
            intent = self.intent_engine.parse(transcript, speaker=role)

            utterances.append({
                "chunk_idx": chunk_idx,
                "timestamp": timestamp,
                "speaker_id": spk_id,
                "role": role,
                "transcript": transcript,
                "intent": intent,
                "separation_ms": sep_ms,
                "asr_ms": asr_ms,
                "total_ms": (time.perf_counter() - t_start) * 1000,
                "is_silent": asr_result.get("is_silent", False),
            })

        return utterances

    def _classify_speakers(
        self,
        sources: torch.Tensor,
        multichannel_mix: torch.Tensor,
    ) -> dict[int, str]:
        """Classify speaker roles (delegates to configured classifier)."""
        method = self.cfg.speaker.method
        if method == "rule":
            return self.speaker_classifier.rule.classify(sources, multichannel_mix)
        else:
            return self.speaker_classifier.classify(sources, multichannel_mix)

    def _should_fallback_separation(
        self,
        mixture: torch.Tensor,
        sources: torch.Tensor,
    ) -> bool:
        """Check if separation quality is too low (SI-SNRi heuristic).

        Note: For CPU-mode (ChannelSelectSeparator), this check is skipped
        because channel selection always produces valid, non-identical outputs.
        """
        # If using CPU fallback, skip quality check (channel selection is always valid)
        sep_cfg = self.cfg.separation
        device = getattr(sep_cfg, "device", "auto")
        has_gpu = torch.cuda.is_available()
        if device == "auto" and not has_gpu:
            return False  # CPU mode, no fallback needed
        if device == "cpu":
            return False

        if sources.shape[0] < 2:
            return True

        energies = (sources ** 2).mean(dim=-1)
        energy_ratio = energies.max() / (energies.min() + 1e-8)
        if energy_ratio < 1.1:
            logger.debug(f"Fallback triggered: energy_ratio={energy_ratio:.2f}")
            return True
        return False

    def _direct_multichannel_asr_sources(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback: use individual microphone channels as "sources"."""
        n_to_use = min(waveform.shape[0], self.cfg.separation.n_speakers)
        return waveform[:n_to_use]  # [N, T]

    def benchmark_memory(
        self,
        wav_paths: list[str | Path],
    ) -> dict[str, float]:
        """Measure peak memory usage over multiple files.

        Parameters
        ----------
        wav_paths : list of paths

        Returns
        -------
        memory_stats : dict with "peak_mb" and "current_mb".
        """
        tracemalloc.start()
        for path in wav_paths:
            try:
                self.process_file(path)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "peak_mb": peak / 1024 / 1024,
            "current_mb": current / 1024 / 1024,
        }
