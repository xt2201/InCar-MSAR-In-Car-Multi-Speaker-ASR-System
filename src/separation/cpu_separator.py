"""Lightweight CPU-friendly speech separation.

Provides two methods that work without GPU/heavy ML models:
1. ChannelSelectSeparator – picks the N channels with highest SNR as "sources"
2. ICASeparator – uses FastICA (sklearn) for blind source separation

These are intended as CPU fallbacks when SepFormer is unavailable or too slow.
Performance is lower than SepFormer but sufficient for demo/test on CPU.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from loguru import logger


class ChannelSelectSeparator:
    """Channel-selection based separation.

    Strategy: sort the 4 microphone channels by RMS energy (highest = closest
    speaker). Select the top-N channels as pseudo-separated sources.
    Each selected channel is treated as one speaker's dominant signal.

    This is a zero-ML baseline that runs instantly on CPU.

    Parameters
    ----------
    n_speakers : int
        Number of sources to output.
    """

    def __init__(self, n_speakers: int = 2) -> None:
        self.n_speakers = n_speakers
        logger.info(f"ChannelSelectSeparator initialized: n_speakers={n_speakers}")

    def separate(
        self,
        multichannel: torch.Tensor,
    ) -> dict:
        """Select top-N channels as separated sources.

        Parameters
        ----------
        multichannel : torch.Tensor
            Shape [C, T] where C = number of channels.

        Returns
        -------
        result : dict with keys "sources" [N, T], "inference_time_ms", "n_speakers"
        """
        t0 = time.perf_counter()

        if multichannel.dim() == 1:
            multichannel = multichannel.unsqueeze(0)

        # If only 1 channel, duplicate it as 2 "sources"
        if multichannel.shape[0] == 1:
            sources = multichannel.expand(self.n_speakers, -1)
            return {
                "sources": sources.contiguous(),
                "n_speakers": self.n_speakers,
                "inference_time_ms": (time.perf_counter() - t0) * 1000,
            }

        # Compute RMS energy per channel
        rms = (multichannel ** 2).mean(dim=-1).sqrt()  # [C]
        # Sort channels by energy descending
        sorted_idx = rms.argsort(descending=True)

        n = min(self.n_speakers, multichannel.shape[0])
        selected = multichannel[sorted_idx[:n]]  # [N, T]

        ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"ChannelSelect: selected channels {sorted_idx[:n].tolist()}, {ms:.1f}ms")

        return {
            "sources": selected.contiguous(),
            "n_speakers": n,
            "inference_time_ms": ms,
        }


class ICASeparator:
    """FastICA-based blind source separation (CPU).

    Uses sklearn FastICA to separate N sources from C microphone channels.
    Works best when N ≤ C and the mixing is reasonably instantaneous.

    Parameters
    ----------
    n_speakers : int
        Number of independent sources to estimate.
    max_iter : int
        Maximum ICA iterations.
    """

    def __init__(self, n_speakers: int = 2, max_iter: int = 200) -> None:
        self.n_speakers = n_speakers
        self.max_iter = max_iter
        logger.info(f"ICASeparator initialized: n_speakers={n_speakers}")

    def separate(self, multichannel: torch.Tensor) -> dict:
        """Separate sources using FastICA.

        Parameters
        ----------
        multichannel : torch.Tensor
            Shape [C, T].

        Returns
        -------
        result : dict with "sources" [N, T], "inference_time_ms".
        """
        try:
            from sklearn.decomposition import FastICA
        except ImportError:
            logger.warning("scikit-learn not installed. Falling back to ChannelSelect.")
            return ChannelSelectSeparator(self.n_speakers).separate(multichannel)

        t0 = time.perf_counter()

        if multichannel.dim() == 1:
            multichannel = multichannel.unsqueeze(0)

        C, T = multichannel.shape
        n = min(self.n_speakers, C)

        audio_np = multichannel.numpy().T  # [T, C]

        try:
            ica = FastICA(n_components=n, max_iter=self.max_iter, random_state=42)
            sources_np = ica.fit_transform(audio_np)  # [T, N]
            sources = torch.tensor(sources_np.T, dtype=torch.float32)  # [N, T]

            # Normalize each source
            for i in range(sources.shape[0]):
                max_val = sources[i].abs().max()
                if max_val > 0:
                    sources[i] = sources[i] / max_val

        except Exception as e:
            logger.warning(f"FastICA failed ({e}). Falling back to ChannelSelect.")
            return ChannelSelectSeparator(n).separate(multichannel)

        ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"ICA separation: {n} sources, {ms:.1f}ms")

        return {
            "sources": sources,
            "n_speakers": n,
            "inference_time_ms": ms,
        }


def get_separator(
    method: str = "channel_select",
    n_speakers: int = 2,
    model_name: str = "sepformer-wsj02mix",
    model_hub: str = "speechbrain/sepformer-wsj02mix",
    device: str = "auto",
):
    """Factory function: return appropriate separator for device.

    Parameters
    ----------
    method : str
        "sepformer" – full SepFormer (GPU recommended)
        "channel_select" – CPU-friendly energy-based selection
        "ica" – FastICA blind source separation
        "auto" – use SepFormer on GPU, channel_select on CPU

    Returns
    -------
    separator : separator instance with .separate(multichannel) method
    """
    import torch
    has_gpu = torch.cuda.is_available()

    if method == "auto":
        method = "sepformer" if has_gpu else "channel_select"

    if method == "sepformer":
        from src.separation.separator import SpeechSeparator
        return SpeechSeparator(
            model_name=model_name,
            model_hub=model_hub,
            n_speakers=n_speakers,
            device=device,
        )
    elif method == "ica":
        return ICASeparator(n_speakers=n_speakers)
    else:
        return ChannelSelectSeparator(n_speakers=n_speakers)
