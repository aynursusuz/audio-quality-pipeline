from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import blake2s, sha256
from pathlib import Path
from typing import Any, Protocol

from .io import write_jsonl

UTMOSV2_WEIGHTS_REVISION = "506474f2b33dc77c234d668cc419be1861899cad"
ECAPA_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"


@dataclass(frozen=True)
class MetricTask:
    audio_id: str
    path: Path
    metric: str
    device: str
    expected_text: str | None = None
    enrollment_path: Path | None = None


class MetricRunner(Protocol):
    model_name: str
    model_revision: str
    model_artifact_sha256: str | None

    def score(self, task: MetricTask, sample_seed: int) -> dict[str, float | str]: ...


class SileroVadRunner:
    model_name = "snakers4/silero-vad"
    model_revision = "package-managed"
    model_artifact_sha256 = None

    def __init__(self) -> None:
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
        except ImportError as exc:
            raise RuntimeError("Install the VAD dependency: pip install '.[vad]'") from exc
        self._timestamps = get_speech_timestamps
        self._read_audio = read_audio
        self._model = load_silero_vad()

    def score(self, task: MetricTask, sample_seed: int) -> dict[str, float | str]:
        del sample_seed
        audio = self._read_audio(task.path.as_posix(), sampling_rate=16000)
        spans = self._timestamps(audio, self._model, return_seconds=True)
        duration_s = len(audio) / 16000
        speech_s = sum(float(span["end"] - span["start"]) for span in spans)
        return {
            "speech_duration_s": round(speech_s, 4),
            "speech_ratio": round(speech_s / duration_s, 6) if duration_s else 0.0,
            "speech_segments": float(len(spans)),
        }


class FasterWhisperRunner:
    model_revision = "runtime-managed"
    model_artifact_sha256 = None

    def __init__(self, model_id: str, cache_dir: Path, device: str) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Install the ASR dependency: pip install '.[asr]'") from exc
        self.model_name = model_id
        compute_type = "float16" if device == "gpu" else "int8"
        self._model = WhisperModel(
            model_id,
            device="cuda" if device == "gpu" else "cpu",
            compute_type=compute_type,
            download_root=(cache_dir / "asr").as_posix(),
        )

    def score(self, task: MetricTask, sample_seed: int) -> dict[str, float | str]:
        del sample_seed
        segments, info = self._model.transcribe(
            task.path.as_posix(), beam_size=1, vad_filter=True, condition_on_previous_text=False
        )
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        values: dict[str, float | str] = {
            "detected_language": info.language,
            "language_probability": round(float(info.language_probability), 6),
            "transcript_characters": float(len(transcript)),
        }
        if task.expected_text is not None:
            values["normalized_wer"] = round(_word_error_rate(task.expected_text, transcript), 6)
        return values


class DnsMosRunner:
    model_name = "microsoft/DNSMOS"
    model_revision = "torchmetrics-managed"
    model_artifact_sha256 = None

    def __init__(self, device: str) -> None:
        self._device = "cuda:0" if device == "gpu" else "cpu"
        try:
            import soundfile
            import torch
            from torchmetrics.functional.audio.dnsmos import (
                deep_noise_suppression_mean_opinion_score,
            )
        except ImportError as exc:
            raise RuntimeError("Install the DNSMOS dependency: pip install '.[dnsmos]'") from exc
        self._soundfile = soundfile
        self._torch = torch
        self._score = deep_noise_suppression_mean_opinion_score

    def score(self, task: MetricTask, sample_seed: int) -> dict[str, float | str]:
        del sample_seed
        audio, sample_rate = self._soundfile.read(
            task.path.as_posix(), dtype="float32", always_2d=True
        )
        mono = audio.mean(axis=1)
        values = self._score(self._torch.from_numpy(mono), sample_rate, False, device=self._device)
        p808, signal, background, overall = (float(value) for value in values.tolist())
        return {
            "p808_mos": p808,
            "signal_mos": signal,
            "background_mos": background,
            "overall_mos": overall,
        }


class UtmosV2Runner:
    model_name = "sarulab-speech/UTMOSv2"
    model_revision = UTMOSV2_WEIGHTS_REVISION

    def __init__(self, checkpoint: Path, device: str) -> None:
        if not checkpoint.is_file():
            raise ValueError(f"UTMOSv2 checkpoint is not a file: {checkpoint}")
        self.model_artifact_sha256 = _file_sha256(checkpoint)
        try:
            import utmosv2
        except ImportError as exc:
            raise RuntimeError("Install UTMOSv2 in the metric worker image") from exc
        self._model = utmosv2.create_model(
            pretrained=True, checkpoint_path=checkpoint, device=_torch_device(device)
        )

    def score(self, task: MetricTask, sample_seed: int) -> dict[str, float | str]:
        import numpy as np

        previous_state = np.random.get_state()
        np.random.seed(sample_seed)
        try:
            score = float(self._model.predict(input_path=task.path.as_posix(), verbose=False))
        finally:
            np.random.set_state(previous_state)
        return {"predicted_mos": score}


class EcapaRunner:
    model_name = "speechbrain/spkrec-ecapa-voxceleb"
    model_revision = ECAPA_REVISION
    model_artifact_sha256 = None

    def __init__(self, cache_dir: Path, device: str) -> None:
        try:
            from huggingface_hub import snapshot_download
            from speechbrain.inference.speaker import SpeakerRecognition
        except ImportError as exc:
            raise RuntimeError("Install the speaker dependency: pip install '.[speaker]'") from exc
        source_dir = snapshot_download(
            repo_id=self.model_name, revision=self.model_revision, local_dir=cache_dir / "source"
        )
        self._model = SpeakerRecognition.from_hparams(
            source=source_dir,
            savedir=(cache_dir / "runtime").as_posix(),
            run_opts={"device": _torch_device(device)},
        )

    def score(self, task: MetricTask, sample_seed: int) -> dict[str, float | str]:
        del sample_seed
        if task.enrollment_path is None:
            raise ValueError("speaker_similarity requires a consented enrollment audio path")
        score, _ = self._model.verify_files(task.enrollment_path.as_posix(), task.path.as_posix())
        return {"speaker_similarity": float(score.squeeze())}


def load_enrollments(path: Path) -> dict[str, Path]:
    values: dict[str, Path] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    value = json.loads(line)
                    values[str(value["audio_id"])] = Path(value["enrollment_path"])
                except (KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid private mapping at {path}:{line_no}: {exc}") from exc
    return values


def load_reference_texts(path: Path) -> dict[str, str]:
    return _load_private_mapping(path, "expected_text")


def _load_private_mapping(path: Path, value_key: str) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    value = json.loads(line)
                    values[str(value["audio_id"])] = str(value[value_key])
                except (KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid private mapping at {path}:{line_no}: {exc}") from exc
    return values


def execute_metric_jobs(
    decisions_path: Path,
    output_path: Path,
    requested_metrics: tuple[str, ...],
    utmos_checkpoint: Path | None,
    enrollments: dict[str, Path],
    model_cache: Path,
    reference_texts: dict[str, str] | None = None,
    asr_model: str = "distil-small.en",
) -> None:
    runners: dict[tuple[str, str], MetricRunner] = {}
    reference_texts = reference_texts or {}

    def results() -> Iterator[dict[str, Any]]:
        for task in iter_metric_tasks(
            decisions_path, requested_metrics, enrollments, reference_texts
        ):
            key = (task.metric, task.device)
            runner = runners.get(key)
            if runner is None:
                runner = _create_runner(
                    task.metric, task.device, utmos_checkpoint, model_cache, asr_model
                )
                runners[key] = runner
            try:
                started = time.perf_counter()
                values = runner.score(task, _stable_seed(task.audio_id))
                result = {
                    "audio_id": task.audio_id,
                    "device": task.device,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "metric": task.metric,
                    "model_name": runner.model_name,
                    "model_revision": runner.model_revision,
                    "status": "ok",
                    "values": values,
                }
                if runner.model_artifact_sha256:
                    result["model_artifact_sha256"] = runner.model_artifact_sha256
                yield result
            except (OSError, RuntimeError, ValueError) as exc:
                yield {
                    "audio_id": task.audio_id,
                    "device": task.device,
                    "error_type": type(exc).__name__,
                    "metric": task.metric,
                    "model_name": runner.model_name,
                    "model_revision": runner.model_revision,
                    "status": "error",
                }

    write_jsonl(output_path, results())


def iter_metric_tasks(
    path: Path,
    requested: tuple[str, ...],
    enrollments: dict[str, Path] | None = None,
    reference_texts: dict[str, str] | None = None,
) -> Iterator[MetricTask]:
    enrollments, reference_texts = enrollments or {}, reference_texts or {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    decision = json.loads(line)
                    for job in decision.get("metric_jobs", []):
                        metric = str(job["metric"])
                        if metric in requested:
                            audio_id = str(decision["audio_id"])
                            yield MetricTask(
                                audio_id,
                                Path(decision["path"]),
                                metric,
                                str(job["device"]),
                                expected_text=reference_texts.get(audio_id)
                                if metric == "asr"
                                else None,
                                enrollment_path=enrollments.get(audio_id)
                                if metric == "speaker_similarity"
                                else None,
                            )
                except (KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid decision record at {path}:{line_no}: {exc}") from exc


def _create_runner(
    metric: str, device: str, checkpoint: Path | None, cache: Path, asr_model: str
) -> MetricRunner:
    if metric == "vad":
        return SileroVadRunner()
    if metric == "asr":
        return FasterWhisperRunner(asr_model, cache, device)
    if metric == "dnsmos":
        return DnsMosRunner(device)
    if metric == "mos":
        if checkpoint is None:
            raise ValueError("--utmos-checkpoint is required when executing mos")
        return UtmosV2Runner(checkpoint, device)
    if metric == "speaker_similarity":
        return EcapaRunner(cache / "ecapa", device)
    raise ValueError(f"No executable worker is registered for metric: {metric}")


def _word_error_rate(reference: str, hypothesis: str) -> float:
    expected, actual = _words(reference), _words(hypothesis)
    if not expected:
        return 0.0 if not actual else 1.0
    previous = list(range(len(actual) + 1))
    for row, word in enumerate(expected, 1):
        current = [row]
        for column, observed in enumerate(actual, 1):
            current.append(
                min(
                    current[-1] + 1, previous[column] + 1, previous[column - 1] + (word != observed)
                )
            )
        previous = current
    return previous[-1] / len(expected)


def _words(value: str) -> list[str]:
    return re.findall(r"\w+", value.casefold())


def _torch_device(device: str) -> str:
    return "cuda:0" if device == "gpu" else device


def _stable_seed(audio_id: str) -> int:
    return int.from_bytes(blake2s(audio_id.encode(), digest_size=4).digest(), "big")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
