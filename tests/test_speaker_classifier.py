"""Unit tests for RuleBasedRoleClassifier."""
import pytest
import torch
from src.speaker.classifier import RuleBasedRoleClassifier


@pytest.fixture
def classifier():
    return RuleBasedRoleClassifier(driver_channel=0)


class TestRuleBasedClassifier:
    def test_two_speakers_returns_driver(self, classifier):
        # Source 0 correlates more with channel 0 (driver)
        driver_signal = torch.sin(torch.linspace(0, 10, 16000))
        passenger_signal = torch.cos(torch.linspace(0, 7, 16000))

        sources = torch.stack([driver_signal, passenger_signal])
        # 4-channel mix where ch0 = driver signal dominant
        mix = torch.stack([
            driver_signal * 0.9 + passenger_signal * 0.1,
            passenger_signal,
            passenger_signal,
            passenger_signal,
        ])

        roles = classifier.classify(sources, mix)
        assert 0 in roles or 1 in roles
        assert "Driver" in roles.values()
        assert any("Passenger" in v for v in roles.values())

    def test_role_map_keys_are_ints(self, classifier):
        sources = torch.randn(2, 8000)
        mix = torch.randn(4, 8000)
        roles = classifier.classify(sources, mix)
        assert all(isinstance(k, int) for k in roles.keys())

    def test_n_speakers_keys_match(self, classifier):
        n = 3
        sources = torch.randn(n, 8000)
        mix = torch.randn(4, 8000)
        roles = classifier.classify(sources, mix)
        assert len(roles) == n

    def test_exactly_one_driver(self, classifier):
        sources = torch.randn(2, 8000)
        mix = torch.randn(4, 8000)
        roles = classifier.classify(sources, mix)
        drivers = [v for v in roles.values() if v == "Driver"]
        assert len(drivers) == 1

    def test_mock_energy_classification(self, classifier):
        roles = classifier.classify_by_energy(
            torch.randn(4, 16000), n_speakers=2
        )
        assert roles == {0: "Driver", 1: "Passenger_1"}
