"""Unit tests for evaluation metrics."""
import pytest
import numpy as np
from src.evaluation.metrics import (
    assign_references_to_hypotheses,
    compute_wer,
    compute_cpwer,
    compute_speaker_accuracy,
    _simple_wer,
)


class TestWER:
    def test_perfect_match(self):
        wer = compute_wer("你好世界", "你好世界")
        assert wer == pytest.approx(0.0, abs=0.01)

    def test_completely_wrong(self):
        wer = compute_wer("abcd", "你好世界")
        assert wer > 0.5

    def test_empty_hypothesis(self):
        wer = compute_wer("", "你好世界")
        assert wer == pytest.approx(1.0, abs=0.01)

    def test_empty_reference(self):
        # Empty reference: WER = 0 if hyp also empty, else infinite
        wer = compute_wer("", "")
        assert wer == 0.0

    def test_partial_match(self):
        wer = compute_wer("你好", "你好世界")
        assert 0 < wer < 1.0


class TestCpWER:
    def test_two_speakers_correct_order(self):
        hyps = ["你好", "再见"]
        refs = ["你好", "再见"]
        cpwer = compute_cpwer(hyps, refs)
        assert cpwer == pytest.approx(0.0, abs=0.01)

    def test_two_speakers_swapped(self):
        """cpWER should find optimal permutation (swapped is same as correct)."""
        hyps = ["再见", "你好"]
        refs = ["你好", "再见"]
        cpwer = compute_cpwer(hyps, refs)
        # cpWER should match swapped order correctly
        assert cpwer == pytest.approx(0.0, abs=0.01)

    def test_empty_inputs(self):
        assert compute_cpwer([], []) == 1.0

    def test_single_speaker(self):
        cpwer = compute_cpwer(["你好世界"], ["你好世界"])
        assert cpwer == pytest.approx(0.0, abs=0.01)

    def test_three_speakers(self):
        hyps = ["音乐", "温度", "导航"]
        refs = ["音乐", "温度", "导航"]
        cpwer = compute_cpwer(hyps, refs)
        assert cpwer == pytest.approx(0.0, abs=0.01)


class TestAssignReferences:
    def test_swapped_hyps_get_optimal_ref(self):
        # refs = SPK1, SPK2 order; hyps are track-ordered and swapped
        refs = ["你好", "再见"]
        hyps = ["再见", "你好"]
        matched = assign_references_to_hypotheses(refs, hyps)
        assert matched == ["再见", "你好"]
        assert compute_wer(hyps[0], matched[0]) + compute_wer(hyps[1], matched[1]) < 0.01

    def test_baseline_one_hyp(self):
        matched = assign_references_to_hypotheses(["A", "B"], ["A"])
        assert len(matched) == 1
        assert matched[0] in ("A", "B")


class TestSpeakerAccuracy:
    def test_all_correct(self):
        pred = ["Driver", "Passenger_1"]
        gt = ["Driver", "Passenger_1"]
        assert compute_speaker_accuracy(pred, gt) == 1.0

    def test_all_wrong(self):
        pred = ["Passenger_1", "Driver"]
        gt = ["Driver", "Passenger_1"]
        assert compute_speaker_accuracy(pred, gt) == 0.0

    def test_partial_correct(self):
        pred = ["Driver", "Driver"]
        gt = ["Driver", "Passenger_1"]
        assert compute_speaker_accuracy(pred, gt) == 0.5

    def test_empty(self):
        assert compute_speaker_accuracy([], []) == 0.0

    def test_case_insensitive(self):
        pred = ["driver", "passenger_1"]
        gt = ["Driver", "Passenger_1"]
        assert compute_speaker_accuracy(pred, gt) == 1.0
