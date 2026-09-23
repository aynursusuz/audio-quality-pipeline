from __future__ import annotations

import argparse
from pathlib import Path

from .metric_runner import execute_metric_jobs, load_enrollments
from .metrics import parse_metric_selection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute selected L2 MOS and speaker-similarity jobs from a decision JSONL"
    )
    parser.add_argument("--decisions", required=True, type=Path, help="L0/L1 decision JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Metric result JSONL")
    parser.add_argument(
        "--metrics",
        required=True,
        help="Explicitly selected metric(s): mos,speaker_similarity",
    )
    parser.add_argument(
        "--utmos-checkpoint",
        type=Path,
        help="Locally provisioned, revision-pinned UTMOSv2 checkpoint required for mos",
    )
    parser.add_argument(
        "--enrollment-manifest",
        type=Path,
        help="Private JSONL mapping audio_id to consented enrollment_path for speaker similarity",
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path(".model-cache"),
        help="Private model cache; do not commit it",
    )
    args = parser.parse_args()
    try:
        requested = parse_metric_selection(args.metrics)
    except ValueError as exc:
        parser.error(str(exc))
    unsupported = sorted(set(requested) - {"mos", "speaker_similarity"})
    if unsupported:
        parser.error(
            "This worker executes only mos and speaker_similarity; "
            f"use dedicated workers for: {', '.join(unsupported)}"
        )
    if "mos" in requested and args.utmos_checkpoint is None:
        parser.error("--utmos-checkpoint is required when executing mos")
    if "speaker_similarity" in requested and args.enrollment_manifest is None:
        parser.error("--enrollment-manifest is required when executing speaker_similarity")

    execute_metric_jobs(
        decisions_path=args.decisions,
        output_path=args.output,
        requested_metrics=requested,
        utmos_checkpoint=args.utmos_checkpoint,
        enrollments=load_enrollments(args.enrollment_manifest)
        if args.enrollment_manifest
        else {},
        model_cache=args.model_cache,
    )


if __name__ == "__main__":
    main()
