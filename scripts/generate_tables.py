#!/usr/bin/env python3
"""Generate LaTeX result tables from evaluation CSV outputs.

Usage:
    python scripts/generate_tables.py
    python scripts/generate_tables.py --output-dir paper/tables
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics-dir", default="outputs/metrics")
    p.add_argument("--output-dir", default="outputs/tables")
    return p.parse_args()


def load_summary(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def format_pct(v, fallback="—") -> str:
    if v is None or (isinstance(v, float) and v != v):  # nan check
        return fallback
    return f"{v*100:.1f}"


def generate_main_results_table(metrics_dir: Path, output_path: Path) -> None:
    """Generate main results table comparing baseline vs pipeline."""
    baseline = load_summary(metrics_dir / "summary_baseline_eval1.json")
    pipeline  = load_summary(metrics_dir / "summary_pipeline_eval1.json")

    upper = load_summary(metrics_dir / "summary_upper_bound_eval1.json")

    rows = [
        {
            "System": "Single-channel ASR (Baseline)",
            "WER (\\%)": format_pct(baseline.get("wer", {}).get("mean")),
            "cpWER (\\%)": format_pct(baseline.get("cpwer", {}).get("mean")),
            "Spk. Acc. (\\%)": "—",
            "Latency (ms/file)": str(baseline.get("latency", {}).get("mean_ms", "—")),
            "Note": "Channel 0, no separation",
        },
        {
            "System": "Multi-channel + Separation + Whisper (Ours)",
            "WER (\\%)": format_pct(pipeline.get("wer", {}).get("mean")),
            "cpWER (\\%)": format_pct(pipeline.get("cpwer", {}).get("mean")),
            "Spk. Acc. (\\%)": "—",
            "Latency (ms/file)": str(pipeline.get("latency", {}).get("mean_ms", "—")),
            "Note": "Full pipeline",
        },
        {
            "System": "Per-channel ASR (Upper-bound proxy)",
            "WER (\\%)": format_pct(upper.get("wer", {}).get("mean")) if upper else "\\textit{N/A}",
            "cpWER (\\%)": format_pct(upper.get("cpwer", {}).get("mean")) if upper else "\\textit{N/A}",
            "Spk. Acc. (\\%)": "\\textit{N/A}",
            "Latency (ms/file)": str(upper.get("latency", {}).get("mean_ms", "—")) if upper else "—",
            "Note": "Each channel → ASR independently",
        },
    ]

    tex = _make_latex_table(
        rows,
        caption="Main Results on AISHELL-5 Eval1. WER and cpWER are lower-is-better. "
                "Near-field upper-bound is unavailable in dev/eval splits.",
        label="tab:main_results",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(tex)
    print(f"Saved: {output_path}")


def generate_ablation_table(metrics_dir: Path, ablation_dir: Path, output_path: Path) -> None:
    """Generate ablation study table."""
    rows = []

    # Full pipeline (from metrics)
    pipeline = load_summary(metrics_dir / "summary_pipeline_eval1.json")
    rows.append({
        "Configuration": "Full Pipeline (Separation + Whisper)",
        "WER (\\%)": format_pct(pipeline.get("wer", {}).get("mean")),
        "cpWER (\\%)": format_pct(pipeline.get("cpwer", {}).get("mean")),
        "Latency (ms/file)": str(pipeline.get("latency", {}).get("mean_ms", "—")),
        "Change": "—",
    })

    # No separation ablation
    no_sep = ablation_dir / "no_separation.csv"
    if no_sep.exists():
        with open(no_sep) as f:
            ablation_rows = list(csv.DictReader(f))
        wers = [float(r["wer"]) for r in ablation_rows if r.get("wer") and r["wer"] != "nan"]
        mean_wer = sum(wers) / len(wers) if wers else float("nan")
        rows.append({
            "Configuration": "w/o Separation (direct ASR on mix)",
            "WER (\\%)": f"{mean_wer*100:.1f}" if mean_wer == mean_wer else "—",
            "cpWER (\\%)": "—",
            "Latency (ms/file)": "—",
            "Change": "$\\uparrow$ WER (worse)",
        })

    # Wav2Vec2 ablation
    w2v_csv = ablation_dir / "wav2vec2_asr.csv"
    if w2v_csv.exists():
        with open(w2v_csv) as f:
            w2v_rows = list(csv.DictReader(f))
        cpwers = [float(r["cpwer"]) for r in w2v_rows if r.get("cpwer") and r["cpwer"] != "nan"]
        mean_cpwer = sum(cpwers) / len(cpwers) if cpwers else float("nan")
        rows.append({
            "Configuration": "Whisper → Wav2Vec2-XLSR (Chinese)",
            "WER (\\%)": "—",
            "cpWER (\\%)": f"{mean_cpwer*100:.1f}" if mean_cpwer == mean_cpwer else "—",
            "Latency (ms/file)": "—",
            "Change": "See discussion",
        })

    tex = _make_latex_table(
        rows,
        caption="Ablation Study on AISHELL-5 Eval1. Each row removes or replaces one component. "
                "Full pipeline is the reference (first row).",
        label="tab:ablation",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(tex)
    print(f"Saved: {output_path}")


def _make_latex_table(rows: list[dict], caption: str, label: str) -> str:
    if not rows:
        return "% No data available\n"

    columns = list(rows[0].keys())
    col_spec = "l" + "c" * (len(columns) - 1)

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{c}}}" for c in columns) + " \\\\",
        "\\midrule",
    ]

    for row in rows:
        lines.append(" & ".join(str(row.get(c, "—")) for c in columns) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    ablation_dir = Path("outputs/ablation")

    output_dir.mkdir(parents=True, exist_ok=True)

    generate_main_results_table(metrics_dir, output_dir / "main_results.tex")
    generate_ablation_table(metrics_dir, ablation_dir, output_dir / "ablation_table.tex")

    print("Table generation complete.")


if __name__ == "__main__":
    main()
