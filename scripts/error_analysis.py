#!/usr/bin/env python3
"""Error analysis script for InCar-MSAR evaluation results.

Reads a WER CSV produced by evaluate.py and generates:
  1. Error breakdown: substitution / insertion / deletion counts per row.
  2. Hallucination detection: rows where hypothesis exhibits repetition loops.
  3. Summary statistics by speaker role.
  4. JSON report suitable for inclusion in a paper appendix.

Usage:
    python scripts/error_analysis.py --csv outputs/metrics/wer_pipeline_eval1.csv
    python scripts/error_analysis.py --csv outputs/metrics/wer_baseline_eval1.csv \\
        --output outputs/metrics/error_analysis_baseline_eval1.json
    python scripts/error_analysis.py --all-csvs outputs/metrics/
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import zlib
from pathlib import Path


# ---------------------------------------------------------------------------
# Edit distance decomposition
# ---------------------------------------------------------------------------

def _edit_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Return (substitutions, insertions, deletions) between ref and hyp.

    Uses standard Levenshtein DP with operation tracking.
    """
    n, m = len(ref), len(hyp)
    # dp[i][j] = (cost, op) where op in {'', 'S', 'I', 'D'}
    INF = n + m + 1

    # Build DP table (cost only, reconstruct by back-tracking)
    # Two rows suffice for cost; store full table for back-track
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
        op[i][0] = "D" if i > 0 else ""
    for j in range(m + 1):
        dp[0][j] = j
        op[0][j] = "I" if j > 0 else ""

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "M"
            else:
                sub = dp[i - 1][j - 1] + 1
                ins = dp[i][j - 1] + 1
                delt = dp[i - 1][j] + 1
                best = min(sub, ins, delt)
                dp[i][j] = best
                if best == sub:
                    op[i][j] = "S"
                elif best == ins:
                    op[i][j] = "I"
                else:
                    op[i][j] = "D"

    # Back-track
    subs = ins = dels = 0
    i, j = n, m
    while i > 0 or j > 0:
        o = op[i][j]
        if o == "S":
            subs += 1
            i -= 1
            j -= 1
        elif o == "I":
            ins += 1
            j -= 1
        elif o == "D":
            dels += 1
            i -= 1
        else:  # "M"
            i -= 1
            j -= 1

    return subs, ins, dels


def edit_ops_chars(ref_str: str, hyp_str: str) -> tuple[int, int, int]:
    """Character-level edit ops. Strip spaces (standard for Mandarin CER)."""
    ref = list(ref_str.replace(" ", ""))
    hyp = list(hyp_str.replace(" ", ""))
    return _edit_ops(ref, hyp)


# ---------------------------------------------------------------------------
# Hallucination detection
# ---------------------------------------------------------------------------

def compression_ratio(text: str) -> float:
    """Estimate zlib compression ratio as a repetition proxy.

    A high ratio indicates many repeated characters (hallucination loop).
    Normal conversation text typically has ratio ≤ 2.0; hallucination
    patterns (e.g. "看完看完看完…") often exceed 4.0.
    """
    if not text:
        return 0.0
    encoded = text.encode("utf-8")
    compressed = zlib.compress(encoded)
    return len(encoded) / max(len(compressed), 1)


def detect_repetition_run(text: str, min_run: int = 5) -> str | None:
    """Return the repeating n-gram if a run of length ≥ min_run is found.

    E.g. "看完看完看完看完看完" → "看完"
    Uses regex for efficiency.
    """
    for ngram_len in range(1, 8):
        pattern = rf"(.{{{ngram_len}}})\1{{{min_run},}}"
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    return None


def is_hallucination(text: str, ratio_threshold: float = 2.4, run_min: int = 5) -> bool:
    """Return True if the text appears to be a Whisper hallucination."""
    if not text:
        return False
    if compression_ratio(text) > ratio_threshold:
        return True
    if detect_repetition_run(text, run_min) is not None:
        return True
    return False


# ---------------------------------------------------------------------------
# Core analysis per row
# ---------------------------------------------------------------------------

def analyse_row(row: dict) -> dict:
    """Compute error breakdown and hallucination flag for a single CSV row."""
    hyp = row.get("hypothesis", "")
    ref = row.get("reference", "")
    wer_raw = row.get("wer", "")

    try:
        wer = float(wer_raw) if wer_raw not in ("", "nan") else float("nan")
    except ValueError:
        wer = float("nan")

    subs, ins, dels = (0, 0, 0)
    ref_len = 0
    if ref:
        subs, ins, dels = edit_ops_chars(ref, hyp)
        ref_len = len(ref.replace(" ", ""))

    total_errors = subs + ins + dels
    hallu = is_hallucination(hyp)
    rep_ngram = detect_repetition_run(hyp)

    return {
        "file_id": row.get("file_id", ""),
        "speaker_id": row.get("speaker_id", ""),
        "role": row.get("role", ""),
        "ref_chars": ref_len,
        "hyp_chars": len(hyp.replace(" ", "")),
        "substitutions": subs,
        "insertions": ins,
        "deletions": dels,
        "total_errors": total_errors,
        "wer": round(wer, 4) if wer == wer else None,
        "compression_ratio": round(compression_ratio(hyp), 3),
        "is_hallucination": hallu,
        "repetition_ngram": rep_ngram,
        "sep_method": row.get("sep_method", ""),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(analysis_rows: list[dict]) -> dict:
    """Aggregate per-row analysis into summary statistics."""
    import math

    def _mean(vals: list[float]) -> float:
        clean = [v for v in vals if v is not None and not math.isnan(v)]
        return sum(clean) / max(1, len(clean))

    total = len(analysis_rows)
    n_hallu = sum(1 for r in analysis_rows if r["is_hallucination"])
    wers = [r["wer"] for r in analysis_rows if r["wer"] is not None]
    subs_total = sum(r["substitutions"] for r in analysis_rows)
    ins_total = sum(r["insertions"] for r in analysis_rows)
    dels_total = sum(r["deletions"] for r in analysis_rows)
    ref_total = sum(r["ref_chars"] for r in analysis_rows)

    # Normalised error rates (proportion of total reference characters)
    ref_safe = max(ref_total, 1)
    sub_rate = subs_total / ref_safe
    ins_rate = ins_total / ref_safe
    del_rate = dels_total / ref_safe

    # By role
    by_role: dict[str, dict] = {}
    for r in analysis_rows:
        role = r.get("role") or "unknown"
        if role not in by_role:
            by_role[role] = {"wers": [], "n_hallu": 0, "n": 0}
        by_role[role]["n"] += 1
        if r["wer"] is not None:
            by_role[role]["wers"].append(r["wer"])
        if r["is_hallucination"]:
            by_role[role]["n_hallu"] += 1

    role_summary = {
        role: {
            "n": v["n"],
            "n_hallucination": v["n_hallu"],
            "mean_wer": round(_mean(v["wers"]), 4),
        }
        for role, v in by_role.items()
    }

    # Hallucination examples
    hallu_examples = [
        {
            "file_id": r["file_id"],
            "repetition_ngram": r["repetition_ngram"],
            "compression_ratio": r["compression_ratio"],
            "hyp_chars": r["hyp_chars"],
        }
        for r in analysis_rows if r["is_hallucination"]
    ][:10]  # top 10

    return {
        "total_rows": total,
        "n_hallucination": n_hallu,
        "hallucination_rate": round(n_hallu / max(total, 1), 4),
        "mean_wer": round(_mean(wers), 4) if wers else None,
        "error_breakdown": {
            "substitutions": subs_total,
            "insertions": ins_total,
            "deletions": dels_total,
            "sub_rate": round(sub_rate, 4),
            "ins_rate": round(ins_rate, 4),
            "del_rate": round(del_rate, 4),
        },
        "by_role": role_summary,
        "hallucination_examples": hallu_examples,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def analyse_csv(csv_path: Path) -> dict:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {"error": "empty CSV", "path": str(csv_path)}

    analysis_rows = [analyse_row(r) for r in rows]
    summary = aggregate(analysis_rows)
    summary["source_csv"] = str(csv_path)
    summary["n_rows_in_csv"] = len(rows)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Error analysis for InCar-MSAR evaluation CSVs")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", type=Path, help="Single WER CSV file to analyse")
    group.add_argument("--all-csvs", type=Path,
                       help="Directory containing WER CSV files (wer_*.csv)")
    p.add_argument("--output", type=Path, default=None,
                   help="Output JSON path (default: same stem as CSV + _error_analysis.json)")
    p.add_argument(
        "--per-row",
        action="store_true",
        help="Include per-row breakdown in the output JSON (verbose)",
    )
    args = p.parse_args()

    results: dict[str, dict] = {}

    if args.csv:
        csv_files = [args.csv]
    else:
        csv_files = sorted(args.all_csvs.glob("wer_*.csv"))

    for csv_path in csv_files:
        print(f"Analysing {csv_path.name}...")
        summary = analyse_csv(csv_path)

        if args.per_row:
            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            summary["per_row"] = [analyse_row(r) for r in rows]

        results[csv_path.name] = summary
        print(
            f"  WER={summary.get('mean_wer', 'n/a'):.1%} | "
            f"hallucination={summary['n_hallucination']}/{summary['total_rows']} "
            f"({summary['hallucination_rate']:.1%}) | "
            f"S/I/D = {summary['error_breakdown']['substitutions']} / "
            f"{summary['error_breakdown']['insertions']} / "
            f"{summary['error_breakdown']['deletions']}"
        )

    # Determine output path
    if args.output:
        out_path = args.output
    elif args.csv:
        out_path = args.csv.with_name(args.csv.stem + "_error_analysis.json")
    else:
        out_path = args.all_csvs / "error_analysis_all.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nError analysis saved → {out_path}")


if __name__ == "__main__":
    main()
