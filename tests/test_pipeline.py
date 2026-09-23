import json
import struct
import sys
import wave
from pathlib import Path

from audio_quality_pipeline.cli import main
from audio_quality_pipeline.dedup import ExactDeduplicator
from audio_quality_pipeline.io import iter_audio_directory, write_jsonl
from audio_quality_pipeline.metric_runner import (
    _file_sha256,
    _stable_seed,
    iter_metric_tasks,
    load_enrollments,
)
from audio_quality_pipeline.metrics import parse_metric_selection, scheduled_metric_jobs
from audio_quality_pipeline.models import AudioRecord
from audio_quality_pipeline.pipeline import inspect_record, iter_feature_extractions
from audio_quality_pipeline.policy import Policy


def _write_tone(
    path: Path, amplitude: float = 0.2, duration_s: float = 1.0, rate: int = 16000
) -> None:
    frames = int(rate * duration_s)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        for index in range(frames):
            # Square wave keeps the fixture standard-library only and deterministic.
            sample = int(32767 * amplitude * (1 if index % 32 < 16 else -1))
            handle.writeframesraw(struct.pack("<h", sample))


def _policy() -> Policy:
    return Policy(
        {
            "version": "test",
            "hard_reject": {
                "min_duration_s": 0.35,
                "max_duration_s": 10.0,
                "min_rms_dbfs": -52.0,
                "max_rms_dbfs": -1.0,
                "max_peak_dbfs": -0.01,
                "max_clipping_ratio": 0.001,
                "max_dc_offset": 0.02,
                "max_silence_ratio": 0.7,
            },
            "review": {
                "min_duration_s": 0.5,
                "max_silence_ratio": 0.5,
                "max_clipping_ratio": 0.0,
                "min_rms_dbfs": -45.0,
            },
            "sampling": {
                "seed": 1,
                "heavy_model_rate": 0.0,
                "human_audit_rate": 0.0,
                "always_heavy_if_review": True,
            },
        }
    )


def test_exact_duplicate_is_rejected(tmp_path: Path) -> None:
    audio = tmp_path / "tone.wav"
    _write_tone(audio)
    policy = _policy()
    policy.document["sampling"]["heavy_model_rate"] = 1.0
    policy.document["sampling"]["human_audit_rate"] = 1.0
    registry = ExactDeduplicator(tmp_path / "dedup.sqlite")
    try:
        first = inspect_record(AudioRecord("a", str(audio)), policy, registry)
        second = inspect_record(AudioRecord("b", str(audio)), policy, registry)
    finally:
        registry.close()
    assert first.status == "accept"
    assert second.status == "reject"
    assert second.duplicate_of == "a"
    assert "hard:exact_duplicate" in second.reasons
    assert not second.route_heavy_model
    assert not second.route_human_audit


def test_silence_is_rejected(tmp_path: Path) -> None:
    audio = tmp_path / "silence.wav"
    _write_tone(audio, amplitude=0.0)
    registry = ExactDeduplicator(tmp_path / "dedup.sqlite")
    try:
        decision = inspect_record(AudioRecord("silence", str(audio)), _policy(), registry)
    finally:
        registry.close()
    assert decision.status == "reject"
    assert "hard:rms_dbfs_below_min" in decision.reasons
    assert not decision.route_heavy_model


def test_policy_rejects_disallowed_sample_rate(tmp_path: Path) -> None:
    audio = tmp_path / "wrong-rate.wav"
    _write_tone(audio, rate=8000)
    policy = _policy()
    policy.document["input"] = {"required_sample_rates_hz": [16000]}
    registry = ExactDeduplicator(tmp_path / "dedup.sqlite")
    try:
        decision = inspect_record(AudioRecord("wrong-rate", str(audio)), policy, registry)
    finally:
        registry.close()
    assert decision.status == "reject"
    assert "hard:sample_rate_not_allowed" in decision.reasons


def test_optional_metrics_are_explicitly_selected_and_device_routed() -> None:
    metric_policy = {"mos": {"device": "gpu"}, "vad": {"device": "cpu"}}
    jobs = scheduled_metric_jobs(
        "accept",
        True,
        parse_metric_selection("mos,vad"),
        metric_policy,
    )
    assert jobs == [
        {"metric": "mos", "device": "gpu", "scope": "routed"},
        {"metric": "vad", "device": "cpu", "scope": "routed"},
    ]


def test_all_expands_to_the_current_metric_catalog() -> None:
    assert parse_metric_selection("all") == (
        "vad",
        "asr",
        "dnsmos",
        "mos",
        "speaker_similarity",
    )


def test_optional_metrics_do_not_run_for_rejected_audio() -> None:
    jobs = scheduled_metric_jobs(
        "reject",
        True,
        parse_metric_selection("mos,speaker_similarity"),
        {},
    )
    assert jobs == []


def test_selected_metric_is_recorded_as_a_job(tmp_path: Path) -> None:
    audio = tmp_path / "tone.wav"
    _write_tone(audio)
    policy = _policy()
    policy.document["sampling"]["heavy_model_rate"] = 1.0
    policy.document["optional_metrics"] = {"mos": {"device": "gpu", "scope": "routed"}}
    registry = ExactDeduplicator(tmp_path / "dedup.sqlite")
    try:
        decision = inspect_record(
            AudioRecord("mos-job", str(audio)), policy, registry, parse_metric_selection("mos")
        )
    finally:
        registry.close()
    assert decision.metric_jobs == [{"metric": "mos", "device": "gpu", "scope": "routed"}]


def test_directory_input_is_stable_and_filters_extensions(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_tone(tmp_path / "b.wav")
    _write_tone(nested / "a.wav")
    (tmp_path / "ignore.txt").write_text("not audio", encoding="utf-8")
    records = list(iter_audio_directory(tmp_path, [".wav"]))
    assert [Path(record.path).relative_to(tmp_path).as_posix() for record in records] == [
        "b.wav",
        "nested/a.wav",
    ]


def test_parallel_extraction_preserves_manifest_order(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    _write_tone(first)
    _write_tone(second)
    records = iter([AudioRecord("first", str(first)), AudioRecord("second", str(second))])
    extractions = list(iter_feature_extractions(records, workers=2, max_in_flight=2))
    assert [extraction.record.audio_id for extraction in extractions] == ["first", "second"]


def test_cli_accepts_directory_input(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "input.wav"
    _write_tone(audio)
    policy = _policy().document
    policy["input"] = {"supported_extensions": [".wav"]}
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    output = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audio-quality-pipeline",
            "--input-dir",
            str(tmp_path),
            "--policy",
            str(policy_path),
            "--dedup-db",
            str(tmp_path / "dedup.sqlite"),
            "--output",
            str(output),
            "--workers",
            "1",
        ],
    )
    main()
    decisions = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(decisions) == 1
    assert decisions[0]["status"] == "accept"


def test_metric_worker_reads_only_selected_scheduled_jobs(tmp_path: Path) -> None:
    decisions = tmp_path / "decisions.jsonl"
    write_jsonl(
        decisions,
        iter(
            [
                {
                    "audio_id": "a",
                    "path": "/private/a.wav",
                    "metric_jobs": [
                        {"metric": "mos", "device": "gpu"},
                        {"metric": "vad", "device": "cpu"},
                    ],
                },
                {"audio_id": "rejected", "path": "/private/rejected.wav"},
            ]
        ),
    )
    tasks = list(iter_metric_tasks(decisions, ("mos",)))
    assert [(task.audio_id, task.metric, task.device) for task in tasks] == [("a", "mos", "gpu")]


def test_enrollment_paths_are_loaded_from_private_mapping(tmp_path: Path) -> None:
    enrollment_path = tmp_path / "enrollments.jsonl"
    enrollment_path.write_text(
        '{"audio_id":"a","enrollment_path":"/private/enrollment.wav"}\n', encoding="utf-8"
    )
    assert load_enrollments(enrollment_path) == {"a": Path("/private/enrollment.wav")}


def test_metric_sampling_seed_is_stable_per_audio_id() -> None:
    assert _stable_seed("audio-a") == _stable_seed("audio-a")
    assert _stable_seed("audio-a") != _stable_seed("audio-b")


def test_model_artifact_hash_is_content_addressed(tmp_path: Path) -> None:
    artifact = tmp_path / "checkpoint.bin"
    artifact.write_bytes(b"checkpoint")
    assert _file_sha256(artifact) == (
        "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef"
    )
