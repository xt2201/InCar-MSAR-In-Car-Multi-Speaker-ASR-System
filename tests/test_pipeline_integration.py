"""Integration tests for the full CPU pipeline.

These tests verify the end-to-end pipeline runs without errors and
produces correctly structured output. They require real AISHELL-5 data
in data/dev/wav/ (run: bash scripts/download_data.sh --dev).

Run: pytest tests/test_pipeline_integration.py -v
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch

# Skip entire module if data/dev/wav/ has no WAV files
DATA_DIR = Path("data/dev")
pytestmark = pytest.mark.skipif(
    not DATA_DIR.exists() or not list(DATA_DIR.glob("wav/*.wav")),
    reason="data/dev/wav/*.wav not found. Run: bash scripts/download_data.sh --dev",
)


@pytest.fixture(scope="module")
def sample_wav():
    """Return path to first WAV in data/dev/wav/ for integration tests."""
    wavs = sorted(DATA_DIR.glob("wav/*.wav"))
    assert wavs, "No WAV files found in data/dev/wav/. Run: bash scripts/download_data.sh --dev"
    return wavs[0]


@pytest.fixture(scope="module")
def waveform_and_sr(sample_wav):
    """Load first WAV from data/dev/wav/."""
    from src.utils.audio import load_multichannel_audio
    wv, sr = load_multichannel_audio(sample_wav)
    return wv, sr


class TestAudioLoading:
    def test_shape_is_4_channel(self, waveform_and_sr):
        wv, sr = waveform_and_sr
        assert wv.shape[0] == 4, f"Expected 4 channels, got {wv.shape[0]}"

    def test_sample_rate_is_16k(self, waveform_and_sr):
        wv, sr = waveform_and_sr
        assert sr == 16000

    def test_duration_at_least_3s(self, waveform_and_sr):
        wv, sr = waveform_and_sr
        dur = wv.shape[-1] / sr
        assert dur >= 3.0, f"Duration {dur:.1f}s < 3s minimum"

    def test_values_normalized(self, waveform_and_sr):
        wv, sr = waveform_and_sr
        assert wv.abs().max() <= 1.0 + 1e-5


class TestCPUSeparation:
    def test_channel_select_runs(self, waveform_and_sr):
        from src.separation.cpu_separator import ChannelSelectSeparator
        wv, sr = waveform_and_sr
        sep = ChannelSelectSeparator(n_speakers=2)
        result = sep.separate(wv)
        assert result["sources"].shape[0] == 2
        assert result["sources"].shape[-1] == wv.shape[-1]
        # 500ms budget: zero-ML, but allow slack for cold-start / loaded systems
        assert result["inference_time_ms"] < 500

    def test_channel_select_output_shape(self, waveform_and_sr):
        from src.separation.cpu_separator import ChannelSelectSeparator
        wv, sr = waveform_and_sr
        sep = ChannelSelectSeparator(n_speakers=3)
        result = sep.separate(wv)
        # Can't exceed n_channels
        assert result["sources"].shape[0] <= 4

    def test_ica_runs(self, waveform_and_sr):
        from src.separation.cpu_separator import ICASeparator
        wv, sr = waveform_and_sr
        sep = ICASeparator(n_speakers=2, max_iter=50)
        result = sep.separate(wv[:, :16000])  # 1s chunk for speed
        assert result["sources"].shape[0] == 2


class TestWhisperCPU:
    @pytest.fixture(scope="class")
    def asr(self):
        import warnings
        warnings.filterwarnings("ignore")
        from src.asr.whisper_asr import WhisperASR
        return WhisperASR(
            model_id="openai/whisper-tiny",
            language="zh",
            device="cpu",
            beam_size=1,
        )

    def test_transcribe_returns_dict(self, asr, waveform_and_sr):
        wv, sr = waveform_and_sr
        result = asr.transcribe(wv[0], sample_rate=sr)
        assert isinstance(result, dict)
        assert "text" in result
        assert "inference_time_ms" in result

    def test_transcribe_returns_string(self, asr, waveform_and_sr):
        wv, sr = waveform_and_sr
        result = asr.transcribe(wv[0], sample_rate=sr)
        assert isinstance(result["text"], str)

    def test_silence_returns_empty(self, asr):
        silence = torch.zeros(16000)
        result = asr.transcribe(silence, sample_rate=16000)
        assert result.get("is_silent") is True
        assert result["text"] == ""

    def test_latency_reasonable_cpu(self, asr, waveform_and_sr):
        """Single 2s chunk should complete in < 10s on CPU."""
        wv, sr = waveform_and_sr
        chunk = wv[0, :32000]  # 2s
        t0 = time.perf_counter()
        asr.transcribe(chunk, sample_rate=sr)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"Inference too slow: {elapsed:.1f}s > 10s limit"


class TestSpeakerClassifier:
    def test_rule_based_returns_driver(self, waveform_and_sr):
        from src.separation.cpu_separator import ChannelSelectSeparator
        from src.speaker.classifier import RuleBasedRoleClassifier
        wv, sr = waveform_and_sr
        sep = ChannelSelectSeparator(n_speakers=2)
        sources = sep.separate(wv)["sources"]
        clf = RuleBasedRoleClassifier(driver_channel=0)
        roles = clf.classify(sources, wv)
        assert "Driver" in roles.values()
        assert len(roles) == 2

    def test_role_labels_valid(self, waveform_and_sr):
        from src.separation.cpu_separator import ChannelSelectSeparator
        from src.speaker.classifier import RuleBasedRoleClassifier
        wv, sr = waveform_and_sr
        sep = ChannelSelectSeparator(n_speakers=2)
        sources = sep.separate(wv)["sources"]
        clf = RuleBasedRoleClassifier()
        roles = clf.classify(sources, wv)
        for role in roles.values():
            assert role in ("Driver", "Passenger_1", "Passenger_2", "Passenger_3")


class TestIntentEngine:
    @pytest.fixture(scope="class")
    def engine(self):
        from src.intent.engine import IntentEngine
        return IntentEngine("configs/intent_keywords.yaml")

    def test_climate_intent(self, engine):
        result = engine.parse("降低温度", "Driver")
        assert result["intent"] == "climate_decrease"

    def test_media_intent(self, engine):
        result = engine.parse("播放音乐", "Driver")
        assert result["intent"] == "media_play"

    def test_unknown_intent(self, engine):
        result = engine.parse("随便说些话", "Driver")
        assert result["intent"] == "unknown"


class TestFullPipelineCPU:
    """End-to-end pipeline test on CPU with real AISHELL-5 data."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        import warnings
        warnings.filterwarnings("ignore")
        from src.pipeline.orchestrator import InCarASRPipeline
        return InCarASRPipeline("configs/default.yaml")

    def test_process_file_returns_list(self, pipeline, sample_wav):
        utterances = pipeline.process_file(sample_wav)
        assert isinstance(utterances, list)
        assert len(utterances) > 0

    def test_output_has_required_fields(self, pipeline, sample_wav):
        utterances = pipeline.process_file(sample_wav)
        required = {"file_id", "speaker_id", "role", "transcript", "intent"}
        for utt in utterances:
            assert required.issubset(utt.keys()), f"Missing keys: {required - utt.keys()}"

    def test_driver_is_assigned(self, pipeline, sample_wav):
        utterances = pipeline.process_file(sample_wav)
        roles = [u["role"] for u in utterances]
        assert "Driver" in roles, f"No Driver found in {roles}"

    def test_intent_field_is_dict(self, pipeline, sample_wav):
        utterances = pipeline.process_file(sample_wav)
        for utt in utterances:
            assert isinstance(utt["intent"], dict)
            assert "intent" in utt["intent"]

    def test_end_to_end_latency(self, pipeline, sample_wav, tmp_path):
        """A short clip should finish in bounded time; avoids multi-minute AISHELL-5 sessions
        in data/dev (materialized) blowing the wall-clock budget."""
        from src.utils.audio import load_multichannel_audio, save_audio

        wv, sr = load_multichannel_audio(sample_wav)
        cap_sec = 5.0
        n = min(wv.shape[1], int(cap_sec * sr))
        wv = wv[:, :n]
        short_wav = tmp_path / "e2e_latency_clip.wav"
        save_audio(wv, short_wav, sample_rate=sr)
        t0 = time.perf_counter()
        pipeline.process_file(short_wav)
        elapsed = time.perf_counter() - t0
        assert elapsed < 120.0, f"Pipeline too slow: {elapsed:.1f}s > 120s for {cap_sec:.0f}s clip"

    def test_multiple_files_no_crash(self, pipeline):
        """Process 5 files without crashing (tests memory/state management)."""
        wavs = sorted(DATA_DIR.glob("wav/*.wav"))[:5]
        for wav in wavs:
            result = pipeline.process_file(wav)
            assert result is not None

    def test_file_not_found_raises(self, pipeline):
        with pytest.raises(FileNotFoundError):
            pipeline.process_file("data/nonexistent/file.wav")
