"""Audio I/O utilities for multi-channel WAV processing.

Uses soundfile as the primary backend (avoids torchaudio codec issues).
Falls back to torchaudio if soundfile is unavailable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


def _load_wav_soundfile(wav_path: str) -> Tuple[np.ndarray, int]:
    """Load WAV using soundfile. Returns (data[T, C], sr)."""
    import soundfile as sf
    data, sr = sf.read(wav_path, always_2d=True)  # [T, C]
    return data, sr


def _resample_numpy(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample [T, C] array from orig_sr to target_sr using scipy."""
    if orig_sr == target_sr:
        return data
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(target_sr, orig_sr)
        up, down = target_sr // g, orig_sr // g
        return resample_poly(data, up, down, axis=0).astype(np.float32)
    except ImportError:
        # Simple linear interpolation fallback
        old_len = data.shape[0]
        new_len = int(old_len * target_sr / orig_sr)
        t_old = np.linspace(0, 1, old_len)
        t_new = np.linspace(0, 1, new_len)
        out = np.zeros((new_len, data.shape[1]), dtype=np.float32)
        for c in range(data.shape[1]):
            out[:, c] = np.interp(t_new, t_old, data[:, c])
        return out


def load_multichannel_audio(
    wav_path: str | Path,
    target_sr: int = 16000,
    normalize: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Load a multi-channel WAV file.

    Parameters
    ----------
    wav_path : str or Path
        Path to the .wav file (expected 4-channel, 16kHz for AISHELL-5).
    target_sr : int
        Target sample rate. Resamples if source differs.
    normalize : bool
        If True, normalize waveform to [-1, 1] per channel.

    Returns
    -------
    waveform : torch.Tensor
        Shape [C, T] where C = number of channels, T = number of samples.
    sample_rate : int
        Effective sample rate after resampling.

    Raises
    ------
    FileNotFoundError
        If wav_path does not exist.
    ValueError
        If audio duration < 0.1 seconds.
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    # Primary: soundfile backend
    try:
        data_np, sr = _load_wav_soundfile(str(wav_path))  # [T, C]
    except Exception:
        # Fallback: torchaudio
        import torchaudio
        wf, sr = torchaudio.load(str(wav_path))  # [C, T]
        data_np = wf.numpy().T  # [T, C]

    # Resample if needed
    if sr != target_sr:
        data_np = _resample_numpy(data_np, sr, target_sr)

    # Convert to [C, T] tensor
    waveform = torch.tensor(data_np.T, dtype=torch.float32)  # [C, T]

    duration = waveform.shape[-1] / target_sr
    if duration < 0.1:
        raise ValueError(f"Audio too short: {duration:.3f}s (min 0.1s required)")

    if normalize:
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val

    return waveform, target_sr


def mix_channels(
    waveform: torch.Tensor,
    weights: Optional[list[float]] = None,
) -> torch.Tensor:
    """Mix multi-channel audio to mono.

    Parameters
    ----------
    waveform : torch.Tensor
        Shape [C, T].
    weights : list of float, optional
        Per-channel mixing weights. Defaults to equal weighting.

    Returns
    -------
    mono : torch.Tensor
        Shape [1, T].
    """
    n_channels = waveform.shape[0]
    if weights is None:
        weights = [1.0 / n_channels] * n_channels

    w = torch.tensor(weights, dtype=waveform.dtype).unsqueeze(1)  # [C, 1]
    mono = (waveform * w).sum(dim=0, keepdim=True)  # [1, T]
    return mono


def save_audio(
    waveform: torch.Tensor,
    path: str | Path,
    sample_rate: int = 16000,
) -> None:
    """Save audio tensor to WAV file.

    Parameters
    ----------
    waveform : torch.Tensor
        Shape [C, T] or [T].
    path : str or Path
        Output file path.
    sample_rate : int
        Sample rate in Hz.
    """
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # [1, T]

    # soundfile expects [T, C]
    audio_np = waveform.cpu().numpy().T
    sf.write(str(path), audio_np, sample_rate, subtype="PCM_16")


def chunk_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    chunk_sec: float = 2.0,
    overlap_sec: float = 0.2,
) -> list[torch.Tensor]:
    """Split audio into overlapping chunks for streaming inference.

    Parameters
    ----------
    waveform : torch.Tensor
        Shape [C, T].
    sample_rate : int
        Sample rate in Hz.
    chunk_sec : float
        Chunk duration in seconds.
    overlap_sec : float
        Overlap duration between consecutive chunks.

    Returns
    -------
    chunks : list of torch.Tensor
        Each chunk has shape [C, chunk_samples].
    """
    chunk_samples = int(chunk_sec * sample_rate)
    overlap_samples = int(overlap_sec * sample_rate)
    step_samples = chunk_samples - overlap_samples
    total_samples = waveform.shape[-1]

    chunks = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[:, start:end]
        # Zero-pad last chunk if shorter than chunk_samples
        if chunk.shape[-1] < chunk_samples:
            pad_size = chunk_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_size))
        chunks.append(chunk)
        start += step_samples

    return chunks


def compute_rms_energy(waveform: torch.Tensor) -> torch.Tensor:
    """Compute RMS energy per channel.

    Parameters
    ----------
    waveform : torch.Tensor
        Shape [C, T].

    Returns
    -------
    energy : torch.Tensor
        Shape [C], RMS energy per channel.
    """
    return (waveform ** 2).mean(dim=-1).sqrt()
