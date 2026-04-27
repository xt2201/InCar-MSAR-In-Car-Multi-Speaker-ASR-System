"""Speech separation module using SpeechBrain SepFormer and Asteroid ConvTasNet."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from loguru import logger


class SpeechSeparator:
    """Multi-speaker speech separation wrapper.

    Supports SepFormer (SpeechBrain) and ConvTasNet (Asteroid).

    Parameters
    ----------
    model_name : str
        "sepformer-wsj02mix" (default) or "convtasnet".
    model_hub : str
        HuggingFace/SpeechBrain model ID.
    n_speakers : int
        Number of speakers to separate (2–4).
    device : str
        "cuda" or "cpu".
    """

    SUPPORTED_MODELS = {
        "sepformer-wsj02mix": "speechbrain",
        "sepformer-whamr": "speechbrain",
        "convtasnet": "asteroid",
    }

    def __init__(
        self,
        model_name: str = "sepformer-wsj02mix",
        model_hub: str = "speechbrain/sepformer-wsj02mix",
        n_speakers: int = 2,
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.model_hub = model_hub
        self.n_speakers = n_speakers
        # "auto" → use CUDA if available, else CPU
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self._model = None

        logger.info(f"SpeechSeparator initialized: model={model_name}, device={self.device}")

    def _load_model(self) -> None:
        """Lazy-load the separation model."""
        if self._model is not None:
            return

        backend = self.SUPPORTED_MODELS.get(self.model_name, "speechbrain")

        if backend == "speechbrain":
            from speechbrain.pretrained import SepformerSeparation
            logger.info(f"Loading SepFormer from {self.model_hub}...")
            self._model = SepformerSeparation.from_hparams(
                source=self.model_hub,
                savedir=f".cache/speechbrain/{self.model_name}",
                run_opts={"device": self.device},
            )
            logger.info("SepFormer loaded successfully.")

        elif backend == "asteroid":
            from asteroid.models import ConvTasNet as AsteroidConvTasNet
            logger.info("Loading ConvTasNet from Asteroid...")
            self._model = AsteroidConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_8k")
            self._model = self._model.to(self.device)
            self._model.eval()
            logger.info("ConvTasNet loaded successfully.")

        else:
            raise ValueError(f"Unsupported model backend: {backend}")

    def separate(
        self,
        mixture: torch.Tensor,
        return_si_snri: bool = False,
    ) -> dict:
        """Separate a monaural mixture into N speaker tracks.

        Parameters
        ----------
        mixture : torch.Tensor
            Mono audio tensor, shape [1, T] or [T]. Normalized to [-1, 1].
        return_si_snri : bool
            If True, compute SI-SNRi (requires reference; returns NaN if unavailable).

        Returns
        -------
        result : dict with keys:
            "sources" : torch.Tensor, shape [N, T] – separated tracks
            "n_speakers" : int
            "inference_time_ms" : float
            "si_snri" : float or None
        """
        self._load_model()

        if mixture.dim() == 1:
            mixture = mixture.unsqueeze(0)

        # Ensure mono input [1, T]
        if mixture.shape[0] > 1:
            mixture = mixture.mean(dim=0, keepdim=True)

        t0 = time.perf_counter()

        with torch.no_grad():
            backend = self.SUPPORTED_MODELS.get(self.model_name, "speechbrain")
            if backend == "speechbrain":
                # SpeechBrain SepFormer expects [T] input
                sources = self._model.separate_batch(mixture.squeeze(0).to(self.device))
                # Output shape: [1, T, N] -> [N, T]
                sources = sources.squeeze(0).permute(1, 0)
            elif backend == "asteroid":
                # Asteroid expects [B, T] input
                inp = mixture.to(self.device)  # [1, T]
                sources = self._model(inp)     # [B, N, T]
                sources = sources.squeeze(0)   # [N, T]

        inference_ms = (time.perf_counter() - t0) * 1000

        # Trim to n_speakers if model outputs more
        sources = sources[:self.n_speakers]

        result = {
            "sources": sources.cpu(),
            "n_speakers": sources.shape[0],
            "inference_time_ms": inference_ms,
            "si_snri": None,
        }

        logger.debug(
            f"Separation done: {sources.shape[0]} sources, "
            f"{inference_ms:.1f}ms, shape={sources.shape}"
        )
        return result

    def compute_si_snri(
        self,
        mixture: torch.Tensor,
        sources: torch.Tensor,
        references: torch.Tensor,
    ) -> float:
        """Compute Scale-Invariant SNR Improvement (SI-SNRi).

        SI-SNRi = SI-SNR(sources, references) - SI-SNR(mixture, references)

        Parameters
        ----------
        mixture : torch.Tensor
            Shape [1, T] or [T].
        sources : torch.Tensor
            Separated sources, shape [N, T].
        references : torch.Tensor
            Clean reference signals, shape [N, T].

        Returns
        -------
        si_snri : float
            Mean SI-SNRi in dB across all speakers.
        """
        def si_snr(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Compute SI-SNR between estimate and target."""
            # Zero-mean
            estimate = estimate - estimate.mean(dim=-1, keepdim=True)
            target = target - target.mean(dim=-1, keepdim=True)
            # Scale factor
            dot = (estimate * target).sum(dim=-1, keepdim=True)
            s_target_energy = (target ** 2).sum(dim=-1, keepdim=True) + 1e-8
            s_target = dot * target / s_target_energy
            e_noise = estimate - s_target
            # SI-SNR in dB
            si_snr_val = 10 * torch.log10(
                (s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + 1e-8)
            )
            return si_snr_val

        if mixture.dim() == 1:
            mixture = mixture.unsqueeze(0)

        # Expand mixture to match N speakers for baseline SI-SNR
        n = references.shape[0]
        mix_expanded = mixture.expand(n, -1)

        si_snr_sep = si_snr(sources[:n], references[:n]).mean().item()
        si_snr_mix = si_snr(mix_expanded, references[:n]).mean().item()
        return si_snr_sep - si_snr_mix


class ChunkedSeparator:
    """Streaming speech separator that processes audio chunk by chunk.

    Uses overlap-add to stitch chunks without artifacts.

    Parameters
    ----------
    separator : SpeechSeparator
        Underlying separator instance.
    chunk_sec : float
        Chunk duration in seconds.
    overlap_sec : float
        Overlap between chunks for smooth stitching.
    sample_rate : int
        Audio sample rate.
    """

    def __init__(
        self,
        separator: SpeechSeparator,
        chunk_sec: float = 2.0,
        overlap_sec: float = 0.2,
        sample_rate: int = 16000,
    ) -> None:
        self.separator = separator
        self.chunk_sec = chunk_sec
        self.overlap_sec = overlap_sec
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_sec * sample_rate)
        self.overlap_samples = int(overlap_sec * sample_rate)
        self.step_samples = self.chunk_samples - self.overlap_samples

    def process_file(
        self,
        waveform: torch.Tensor,
        mix_to_mono: bool = True,
    ) -> dict:
        """Process a full audio file using chunked separation.

        Parameters
        ----------
        waveform : torch.Tensor
            Multi-channel audio [C, T] or mono [1, T].
        mix_to_mono : bool
            If True, mix channels to mono before separation (for SepFormer).
            If False, pass multichannel chunks directly (for CPU separators).

        Returns
        -------
        result : dict with keys:
            "sources" : torch.Tensor [N, T] – full-length separated tracks
            "chunks_processed" : int
            "total_inference_ms" : float
            "per_chunk_latency_ms" : list[float]
        """
        from src.utils.audio import mix_channels, chunk_audio

        # Keep original multichannel for chunking when not mixing to mono
        if mix_to_mono and waveform.shape[0] > 1:
            sep_input = mix_channels(waveform)  # [1, T]
        else:
            sep_input = waveform  # [C, T] for multichannel separation

        chunks = chunk_audio(
            sep_input,
            self.sample_rate,
            self.chunk_sec,
            self.overlap_sec,
        )

        separated_chunks = []
        latencies = []

        logger.info(f"Processing {len(chunks)} chunks ({self.chunk_sec}s each)...")

        for i, chunk in enumerate(chunks):
            result = self.separator.separate(chunk)
            separated_chunks.append(result["sources"])  # [N, chunk_T]
            latencies.append(result["inference_time_ms"])

        # Reconstruct full-length sources using overlap-add
        sources = self._overlap_add(separated_chunks)

        # Trim to original length
        original_len = sep_input.shape[-1]
        sources = sources[:, :original_len]

        return {
            "sources": sources,
            "chunks_processed": len(chunks),
            "total_inference_ms": sum(latencies),
            "per_chunk_latency_ms": latencies,
        }

    def _overlap_add(self, chunks: list[torch.Tensor]) -> torch.Tensor:
        """Stitch separated chunks using overlap-add method.

        Parameters
        ----------
        chunks : list of torch.Tensor
            Each chunk has shape [N, chunk_T].

        Returns
        -------
        output : torch.Tensor
            Shape [N, total_T].
        """
        if not chunks:
            raise ValueError("No chunks to stitch")

        n_speakers = chunks[0].shape[0]
        chunk_len = chunks[0].shape[1]

        total_len = self.step_samples * (len(chunks) - 1) + chunk_len
        output = torch.zeros(n_speakers, total_len)
        weight = torch.zeros(total_len)

        # Hann window for smooth overlap-add
        window = torch.hann_window(chunk_len)

        for i, chunk in enumerate(chunks):
            start = i * self.step_samples
            end = start + chunk_len
            if end > total_len:
                end = total_len
                chunk = chunk[:, : end - start]
                w = window[: end - start]
            else:
                w = window

            output[:, start:end] += chunk * w.unsqueeze(0)
            weight[start:end] += w

        # Normalize by accumulated weights
        weight = weight.clamp(min=1e-8)
        output = output / weight.unsqueeze(0)

        return output
