"""Unit tests for config loading."""
import pytest
from pathlib import Path
from src.utils.config import load_config


def test_load_default_config():
    """Config should load without error."""
    cfg = load_config("configs/default.yaml")
    assert cfg is not None


def test_config_audio_params():
    """Check required audio params exist with valid values."""
    cfg = load_config("configs/default.yaml")
    assert cfg.audio.sample_rate == 16000
    assert cfg.audio.n_channels == 4
    assert cfg.audio.chunk_size_sec > 0
    assert cfg.audio.min_duration_sec > 0


def test_config_separation_params():
    cfg = load_config("configs/default.yaml")
    assert cfg.separation.model is not None
    assert cfg.separation.n_speakers in [2, 3, 4]
    assert cfg.separation.si_snri_threshold >= 0


def test_config_asr_params():
    cfg = load_config("configs/default.yaml")
    assert "whisper" in cfg.asr.model.lower()
    assert cfg.asr.language == "zh"
    assert cfg.asr.beam_size >= 1


def test_config_not_found():
    """Missing config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("configs/nonexistent.yaml")
