from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path

from .dedup import ExactDeduplicator
from .io import stable_sample
from .metrics import scheduled_metric_jobs
from .models import AudioRecord, Decision, SignalFeatures
from .policy import Policy
from .signal import AudioDecodeError, extract_wav_features


@dataclass(frozen=True)
class FeatureExtraction:
    record: AudioRecord
    features: SignalFeatures | None = None
    decode_error: str | None = None


def extract_record(record: AudioRecord) -> FeatureExtraction:
    """Pickle-safe CPU work for a process pool; no policy state is mutated here."""
    try:
        return FeatureExtraction(record=record, features=extract_wav_features(Path(record.path)))
    except AudioDecodeError as exc:
        return FeatureExtraction(record=record, decode_error=str(exc))


def iter_feature_extractions(
    records: Iterator[AudioRecord], workers: int, max_in_flight: int
) -> Iterator[FeatureExtraction]:
    """Parallel extraction with bounded memory and manifest-order output."""
    if workers == 1:
        yield from map(extract_record, records)
        return

    record_iterator = enumerate(records)
    next_output_index = 0
    ready: dict[int, FeatureExtraction] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {}

        def submit_one() -> bool:
            try:
                index, record = next(record_iterator)
            except StopIteration:
                return False
            pending[pool.submit(extract_record, record)] = index
            return True

        for _ in range(max_in_flight):
            if not submit_one():
                break
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                ready[pending.pop(future)] = future.result()
            while next_output_index in ready:
                yield ready.pop(next_output_index)
                next_output_index += 1
                submit_one()


def decide_extraction(
    extraction: FeatureExtraction,
    policy: Policy,
    dedup: ExactDeduplicator,
    requested_metrics: tuple[str, ...] = (),
) -> Decision:
    record = extraction.record
    if extraction.decode_error:
        return Decision(
            audio_id=record.audio_id,
            path=record.path,
            status="reject",
            reasons=["hard:decode_failed"],
            features={"decode_error": extraction.decode_error},
            pipeline_version=policy.version,
        )
    assert extraction.features is not None
    features = extraction.features
    status, reasons = policy.evaluate(features)
    duplicate_of = dedup.first_seen_or_duplicate(features.sha256, record.audio_id)
    if duplicate_of:
        status = "reject"
        reasons = [*reasons, "hard:exact_duplicate"]

    sampling = policy.document["sampling"]
    eligible_for_expensive_review = status != "reject"
    is_review = status == "review"
    route_heavy = eligible_for_expensive_review and (
        (is_review and sampling["always_heavy_if_review"])
        or stable_sample(record.audio_id, sampling["heavy_model_rate"], sampling["seed"])
    )
    route_human = eligible_for_expensive_review and stable_sample(
        record.audio_id, sampling["human_audit_rate"], sampling["seed"]
    )
    return Decision(
        audio_id=record.audio_id,
        path=record.path,
        status=status,
        reasons=reasons,
        features=asdict(features),
        duplicate_of=duplicate_of,
        route_heavy_model=route_heavy,
        route_human_audit=route_human,
        metric_jobs=scheduled_metric_jobs(
            status,
            route_heavy,
            requested_metrics,
            policy.document.get("optional_metrics", {}),
        ),
        pipeline_version=policy.version,
    )


def inspect_record(
    record: AudioRecord,
    policy: Policy,
    dedup: ExactDeduplicator,
    requested_metrics: tuple[str, ...] = (),
) -> Decision:
    return decide_extraction(extract_record(record), policy, dedup, requested_metrics)
