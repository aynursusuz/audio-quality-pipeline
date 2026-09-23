from __future__ import annotations

import argparse
from pathlib import Path

from .dedup import ExactDeduplicator
from .io import iter_audio_directory, iter_manifest, write_jsonl
from .metrics import parse_metric_selection, validate_metric_policy
from .pipeline import decide_extraction, iter_feature_extractions
from .policy import Policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier-0 audio quality control")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path, help="Input JSONL manifest")
    inputs.add_argument("--input-dir", type=Path, help="Directory of supported audio files")
    parser.add_argument("--output", required=True, type=Path, help="Decision JSONL output")
    parser.add_argument("--policy", required=True, type=Path, help="Versioned policy JSON")
    parser.add_argument(
        "--dedup-db", required=True, type=Path, help="Exact duplicate SQLite registry"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="CPU extraction processes (default: 1)"
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        help="Maximum decoded jobs kept in memory (default: workers × 4)",
    )
    parser.add_argument(
        "--metrics",
        default="",
        help="Optional jobs to schedule, comma-separated (for example: mos,speaker_similarity)",
    )
    args = parser.parse_args()
    try:
        requested_metrics = parse_metric_selection(args.metrics)
    except ValueError as exc:
        parser.error(str(exc))

    policy = Policy.from_file(args.policy)
    try:
        validate_metric_policy(requested_metrics, policy.document.get("optional_metrics", {}))
    except ValueError as exc:
        parser.error(str(exc))
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    max_in_flight = args.max_in_flight or args.workers * 4
    if max_in_flight < args.workers:
        parser.error("--max-in-flight must be at least --workers")
    records = (
        iter_manifest(args.manifest)
        if args.manifest
        else iter_audio_directory(args.input_dir, policy.document["input"]["supported_extensions"])
    )
    dedup = ExactDeduplicator(args.dedup_db)
    try:
        write_jsonl(
            args.output,
            (
                decide_extraction(extraction, policy, dedup, requested_metrics).as_dict()
                for extraction in iter_feature_extractions(records, args.workers, max_in_flight)
            ),
        )
    finally:
        dedup.close()


if __name__ == "__main__":
    main()
