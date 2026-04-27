"""Unit tests for audio utilities."""
import pytest
import torch
from src.utils.audio import mix_channels, compute_rms_energy, chunk_audio


class TestMixChannels:
    def test_mix_4ch_to_mono(self):
        audio = torch.randn(4, 16000)
        mono = mix_channels(audio)
        assert mono.shape == (1, 16000)

    def test_mix_1ch_unchanged(self):
        audio = torch.randn(1, 16000)
        mono = mix_channels(audio)
        assert mono.shape == (1, 16000)

    def test_mix_with_weights(self):
        audio = torch.ones(2, 100)
        mono = mix_channels(audio, weights=[0.8, 0.2])
        # Result should be weighted sum
        expected = 0.8 * 1.0 + 0.2 * 1.0
        assert torch.allclose(mono, torch.full((1, 100), expected), atol=1e-5)

    def test_mix_preserves_length(self):
        T = 32000
        audio = torch.randn(4, T)
        mono = mix_channels(audio)
        assert mono.shape[-1] == T


class TestRMSEnergy:
    def test_silence_has_zero_energy(self):
        audio = torch.zeros(4, 16000)
        energy = compute_rms_energy(audio)
        assert (energy == 0.0).all()

    def test_energy_shape(self):
        audio = torch.randn(4, 16000)
        energy = compute_rms_energy(audio)
        assert energy.shape == (4,)

    def test_energy_nonnegative(self):
        audio = torch.randn(4, 16000)
        energy = compute_rms_energy(audio)
        assert (energy >= 0).all()


class TestChunkAudio:
    def test_single_chunk(self):
        sr = 16000
        audio = torch.randn(1, sr * 2)  # 2 seconds
        chunks = chunk_audio(audio, sr, chunk_sec=2.0, overlap_sec=0.0)
        assert len(chunks) == 1
        assert chunks[0].shape == (1, sr * 2)

    def test_multiple_chunks(self):
        sr = 16000
        audio = torch.randn(1, sr * 5)  # 5 seconds
        chunks = chunk_audio(audio, sr, chunk_sec=2.0, overlap_sec=0.0)
        assert len(chunks) >= 2

    def test_chunk_shape(self):
        sr = 16000
        audio = torch.randn(4, sr * 3)  # 4-channel, 3s
        chunks = chunk_audio(audio, sr, chunk_sec=2.0, overlap_sec=0.2)
        chunk_samples = int(2.0 * sr)
        for c in chunks:
            assert c.shape[0] == 4
            assert c.shape[-1] == chunk_samples
