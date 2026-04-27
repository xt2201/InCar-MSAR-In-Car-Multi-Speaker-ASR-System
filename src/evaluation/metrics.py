"""Evaluation metrics for multi-speaker ASR.

Implements:
- CER/WER: Character-level edit distance (standard for Mandarin)
- cpWER: Mean matched CER after Hungarian optimal speaker assignment
- Speaker Attribution Accuracy
- SI-SNRi (Scale-Invariant SNR Improvement)
"""
from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
from loguru import logger


def compute_wer(hypothesis: str, reference: str) -> float:
    """Compute character-level WER between hypothesis and reference.

    For Chinese Mandarin (no word boundaries), we compute Character Error Rate
    (CER) by treating each character as a "word". This is the standard metric
    used in AISHELL benchmarks.

    Parameters
    ----------
    hypothesis : str
        Predicted transcript.
    reference : str
        Ground truth transcript.

    Returns
    -------
    wer : float in [0, 1+] (can exceed 1 for many insertions).
    """
    # For Chinese, use character-level edit distance directly
    # (equivalent to CER, the standard metric for Mandarin ASR)
    return _simple_wer(hypothesis, reference)


def _simple_wer(hypothesis: str, reference: str) -> float:
    """Simple WER fallback without jiwer."""
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))

    if not ref_chars:
        return float(len(hyp_chars) > 0)

    # Edit distance
    d = _edit_distance(ref_chars, hyp_chars)
    return d / len(ref_chars)


def _edit_distance(ref: list, hyp: list) -> int:
    """Levenshtein edit distance."""
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new_dp = [i] + [0] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp
    return dp[m]


def compute_cpwer(
    hypotheses: list[str],
    references: list[str],
) -> float:
    """Compute Concatenated Permutation WER (cpWER) for multi-speaker ASR.

    cpWER finds the permutation of hypotheses that minimizes the total WER
    when compared to references using the Hungarian algorithm.

    Parameters
    ----------
    hypotheses : list of str
        Predicted transcripts for each speaker track (N items).
    references : list of str
        Ground truth transcripts for each speaker (N items).

    Returns
    -------
    cpwer : float
        Minimum WER over all permutations.

    Notes
    -----
    For N speakers, there are N! permutations. Hungarian algorithm
    reduces this to O(N^3) complexity. For N ≤ 4 (AISHELL-5 max),
    brute force over N! ≤ 24 permutations is also acceptable.
    """
    n = len(references)
    m = len(hypotheses)

    if n == 0 or m == 0:
        return 1.0

    # Build cost matrix [n_refs x n_hyps]
    cost_matrix = np.zeros((n, m), dtype=np.float64)
    for i, ref in enumerate(references):
        for j, hyp in enumerate(hypotheses):
            cost_matrix[i, j] = compute_wer(hyp, ref)

    # Hungarian algorithm (minimize total cost = total WER)
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        min_wer = cost_matrix[row_ind, col_ind].mean()
    except ImportError:
        logger.warning("scipy not available. Using brute force cpWER (N≤4).")
        min_wer = _brute_force_cpwer(cost_matrix)

    return float(min_wer)


def assign_references_to_hypotheses(
    references: list[str],
    hypotheses: list[str],
) -> list[str]:
    """Pair each hypothesis to a reference via Hungarian (same cost as cpWER).

    This is the correct per-utterance reference for AISHELL-5: pipeline outputs
    are not ordered as SPK1, SPK2, … — they follow separated track order.

    Parameters
    ----------
    references
        One string per ground-truth speaker, e.g. from sorted SPK1, SPK2, …
    hypotheses
        One ASR output per non-silent source (same order as evaluation rows for hyps).

    Returns
    -------
    matched_refs
        ``len(matched_refs) == len(hypotheses)``. Entry ``j`` is the reference
        string assigned to ``hypotheses[j]`` (empty if unassigned in rectangular case).
    """
    n_r, n_h = len(references), len(hypotheses)
    if n_h == 0:
        return []
    if n_r == 0:
        return [""] * n_h
    cost_matrix = np.zeros((n_r, n_h), dtype=np.float64)
    for i, ref in enumerate(references):
        for j, hyp in enumerate(hypotheses):
            cost_matrix[i, j] = compute_wer(hyp, ref)
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ImportError:
        # Rare: no scipy; pair greedily (not optimal)
        out = [""] * n_h
        used_r = set()
        for j in range(n_h):
            best, bi = 1.0, -1
            for i in range(n_r):
                if i in used_r:
                    continue
                c = cost_matrix[i, j]
                if c < best:
                    best, bi = c, i
            if bi >= 0:
                out[j] = references[bi]
                used_r.add(bi)
        return out

    out = [""] * n_h
    for ri, cj in zip(row_ind, col_ind):
        if 0 <= cj < n_h and 0 <= ri < n_r:
            out[cj] = references[ri]
    return out


def _brute_force_cpwer(cost_matrix: np.ndarray) -> float:
    """Brute force permutation search for small N."""
    n = cost_matrix.shape[0]
    m = cost_matrix.shape[1]
    k = min(n, m)
    best = float("inf")
    for perm in itertools.permutations(range(m), k):
        total = sum(cost_matrix[i, perm[i]] for i in range(k))
        if total < best:
            best = total
    return best / k


def compute_speaker_accuracy(
    predicted_roles: list[str],
    ground_truth_roles: list[str],
) -> float:
    """Compute speaker role attribution accuracy.

    Parameters
    ----------
    predicted_roles : list of str
        Predicted role labels, e.g. ["Driver", "Passenger_1"].
    ground_truth_roles : list of str
        Ground truth role labels.

    Returns
    -------
    accuracy : float in [0, 1].
    """
    if not ground_truth_roles:
        return 0.0

    n = max(len(ground_truth_roles), len(predicted_roles))
    matches = sum(
        1 for p, g in zip(predicted_roles, ground_truth_roles)
        if p.lower() == g.lower()
    )
    return matches / n


def compute_si_snri_batch(
    mixtures: list,
    sources_list: list,
    references_list: list,
) -> list[float]:
    """Compute SI-SNRi for a batch of samples.

    Parameters
    ----------
    mixtures : list of tensors [1, T]
    sources_list : list of tensors [N, T]
    references_list : list of tensors [N, T]

    Returns
    -------
    si_snri_values : list of float (dB)
    """
    from src.separation.separator import SpeechSeparator

    separator = SpeechSeparator.__new__(SpeechSeparator)
    results = []

    for mix, sources, refs in zip(mixtures, sources_list, references_list):
        try:
            val = separator.compute_si_snri(mix, sources, refs)
            results.append(val)
        except Exception as e:
            logger.warning(f"SI-SNRi computation failed: {e}")
            results.append(float("nan"))

    return results


def summarize_metrics(wer_list: list[float], cpwer_list: list[float]) -> dict:
    """Compute summary statistics over a list of metric values.

    Parameters
    ----------
    wer_list : list of WER values per sample
    cpwer_list : list of cpWER values per sample

    Returns
    -------
    summary : dict with mean, median, std, min, max for each metric.
    """
    def stats(values: list[float]) -> dict:
        arr = np.array([v for v in values if not np.isnan(v)])
        if len(arr) == 0:
            return {"mean": float("nan"), "median": float("nan"), "std": float("nan"),
                    "min": float("nan"), "max": float("nan"), "n": 0}
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": len(arr),
        }

    return {
        "wer": stats(wer_list),
        "cpwer": stats(cpwer_list),
    }
