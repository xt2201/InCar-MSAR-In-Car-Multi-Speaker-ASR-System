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


class BeamformSeparator:
    """Delay-and-Sum / MVDR beamforming-based separation for 4-channel audio.

    Exploits spatial information from the 4 microphones to steer towards each
    speaker and suppress interference.  The implementation uses a simplified
    Minimum Variance Distortionless Response (MVDR) beamformer computed from
    the sample covariance matrices of the input signal.

    This CPU-only method requires no GPU and no ML models.  It is significantly
    better than ChannelSelect because it actively uses phase relationships
    between channels, not just energy ranking.

    Parameters
    ----------
    n_speakers : int
        Number of output sources (1–4; must not exceed n_channels).
    frame_sec : float
        Analysis frame duration in seconds (default 0.032 = 32 ms).
    hop_sec : float
        Frame hop in seconds (default 0.016 = 16 ms).
    reg : float
        Regularisation coefficient added to the covariance diagonal to prevent
        rank-deficiency on short or near-silent segments.
    """

    def __init__(
        self,
        n_speakers: int = 2,
        frame_sec: float = 0.032,
        hop_sec: float = 0.016,
        reg: float = 1e-5,
    ) -> None:
        self.n_speakers = n_speakers
        self.frame_sec = frame_sec
        self.hop_sec = hop_sec
        self.reg = reg
        logger.info(f"BeamformSeparator (MVDR) initialized: n_speakers={n_speakers}")

    def separate(self, multichannel: torch.Tensor) -> dict:
        """Apply MVDR beamforming to produce N pseudo-separated sources.

        Parameters
        ----------
        multichannel : torch.Tensor
            Shape [C, T] where C ≥ 2.

        Returns
        -------
        result : dict with "sources" [N, T], "inference_time_ms", "n_speakers"
        """
        t0 = time.perf_counter()

        if multichannel.dim() == 1:
            multichannel = multichannel.unsqueeze(0)

        C, T = multichannel.shape
        n = min(self.n_speakers, C)

        if C < 2:
            # Degenerate: just duplicate the single channel
            sources = multichannel.expand(n, -1).contiguous()
            return {
                "sources": sources,
                "n_speakers": n,
                "inference_time_ms": (time.perf_counter() - t0) * 1000,
            }

        audio_np = multichannel.numpy()  # [C, T]
        sr = 16000  # AISHELL-5 standard

        try:
            sources_np = self._mvdr_beamform(audio_np, sr, n)
        except Exception as e:
            logger.warning(f"MVDR beamforming failed ({e}). Falling back to ChannelSelect.")
            return ChannelSelectSeparator(n).separate(multichannel)

        sources = torch.tensor(sources_np, dtype=torch.float32)  # [N, T]

        ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"BeamformSeparator: {n} sources, {ms:.1f}ms")

        return {
            "sources": sources,
            "n_speakers": n,
            "inference_time_ms": ms,
        }

    def _mvdr_beamform(self, audio: np.ndarray, sr: int, n_out: int) -> np.ndarray:
        """Compute MVDR beamformer outputs for n_out virtual look-directions.

        Strategy:
        1. Estimate the full-band spatial covariance matrix R = X X^H.
        2. Find the n_out principal eigenvectors of R — each corresponds to a
           dominant spatial direction (speaker location).
        3. For each eigenvector v_k, compute the MVDR weight vector:
               w_k = R^{-1} v_k / (v_k^H R^{-1} v_k)
        4. Apply w_k to the multichannel signal to extract source k.

        This is a broadband (single-band) MVDR approximation suitable for
        real-time CPU processing.  A proper narrowband MVDR would operate
        per frequency bin in the STFT domain, giving better spatial resolution
        but at much higher computational cost.
        """
        import numpy as np
        from numpy.linalg import eigh, solve

        C, T = audio.shape
        # Spatial covariance matrix: [C, C] complex (use real here for simplicity)
        R = (audio @ audio.T) / T  # [C, C], real
        R += self.reg * np.eye(C)  # regularise

        # Eigendecomposition: eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = eigh(R)
        # Take top n_out eigenvectors (largest eigenvalues = dominant speakers)
        top_idx = np.argsort(eigenvalues)[::-1][:n_out]
        steering = eigenvectors[:, top_idx]  # [C, n_out]

        sources = np.zeros((n_out, T), dtype=np.float32)
        R_inv = np.linalg.inv(R)  # [C, C]

        for k in range(n_out):
            v = steering[:, k]  # [C]
            # MVDR weight: w = R^{-1} v / (v^H R^{-1} v)
            Rinv_v = R_inv @ v  # [C]
            denom = v @ Rinv_v + 1e-10
            w = Rinv_v / denom  # [C]
            # Beamform
            bf = w @ audio  # [T]
            # Normalise
            peak = np.abs(bf).max()
            if peak > 1e-8:
                bf = bf / peak
            sources[k] = bf.astype(np.float32)

        return sources


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
        "beamform" – CPU MVDR beamforming (exploits 4-channel spatial info)
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
    elif method == "beamform":
        return BeamformSeparator(n_speakers=n_speakers)
    else:
        return ChannelSelectSeparator(n_speakers=n_speakers)
