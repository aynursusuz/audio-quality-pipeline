from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import blake2s, sha256
from pathlib import Path
from typing import Any, Protocol

from .io import write_jsonl

UTMOSV2_CODE_REVISION = "cc2700db57bb83ee13dc31ebe1b868c254e15d09"
UTMOSV2_WEIGHTS_REVISION = "506474f2b33dc77c234d668cc419be1861899cad"
ECAPA_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"


@dataclass(frozen=True)
class MetricTask:
    audio_id: str
    path: Path
    metric: str
    device: str


class MetricRunner(Protocol):
    model_name: str
    model_revision: str
    model_artifact_sha256: str | None

    def score(
        self, audio_path: Path, enrollment_path: Path | None = None, sample_seed: int = 0
    ) -> dict[str, float]: ...


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
            message = (
                "Install the private worker dependency: "
                "pip install git+https://github.com/sarulab-speech/UTMOSv2.git"
            )
            raise RuntimeError(message) from exc
        self._model = utmosv2.create_model(
            pretrained=True, checkpoint_path=checkpoint, device=_torch_device(device)
        )

    def score(
        self, audio_path: Path, enrollment_path: Path | None = None, sample_seed: int = 0
    ) -> dict[str, float]:
        del enrollment_path
        import numpy as np

        previous_state = np.random.get_state()
        np.random.seed(sample_seed)
        try:
            score = float(self._model.predict(input_path=audio_path.as_posix(), verbose=False))
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
            raise RuntimeError(
                "Install the private worker dependency: pip install '.[speaker]'"
            ) from exc
        cache_dir.mkdir(parents=True, exist_ok=True)
        source_dir = snapshot_download(
            repo_id=self.model_name,
            revision=self.model_revision,
            local_dir=cache_dir / "source",
        )
        self._model = SpeakerRecognition.from_hparams(
            source=source_dir,
            savedir=(cache_dir / "runtime").as_posix(),
            run_opts={"device": _torch_device(device)},
        )

    def score(
        self, audio_path: Path, enrollment_path: Path | None = None, sample_seed: int = 0
    ) -> dict[str, float]:
        del sample_seed
        if enrollment_path is None:
            raise ValueError("speaker_similarity requires a consented enrollment audio path")
        score, _ = self._model.verify_files(enrollment_path.as_posix(), audio_path.as_posix())
        return {"speaker_similarity": float(score.squeeze())}


def load_enrollments(path: Path) -> dict[str, Path]:
    """Load private audio_id-to-enrollment-path mappings without persisting them in results."""
    enrollments: dict[str, Path] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                enrollments[str(value["audio_id"])] = Path(value["enrollment_path"])
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid enrollment record at {path}:{line_no}: {exc}") from exc
    return enrollments


def execute_metric_jobs(
    decisions_path: Path,
    output_path: Path,
    requested_metrics: tuple[str, ...],
    utmos_checkpoint: Path | None,
    enrollments: dict[str, Path],
    model_cache: Path,
) -> None:
    """Stream decision jobs and retain no embeddings, transcripts, or audio in result JSONL."""
    runners: dict[tuple[str, str], MetricRunner] = {}

    def results() -> Iterator[dict[str, Any]]:
        for task in iter_metric_tasks(decisions_path, requested_metrics):
            runner = runners.get((task.metric, task.device))
            if runner is None:
                runner = _create_runner(task.metric, task.device, utmos_checkpoint, model_cache)
                runners[(task.metric, task.device)] = runner
            started = time.perf_counter()
            try:
                values = runner.score(
                    task.path,
                    enrollments.get(task.audio_id),
                    _stable_seed(task.audio_id),
                )
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


def iter_metric_tasks(path: Path, requested_metrics: tuple[str, ...]) -> Iterator[MetricTask]:
    """Read only explicitly scheduled jobs; rejected records have no executable job."""
    requested = set(requested_metrics)
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                decision = json.loads(line)
                for job in decision.get("metric_jobs", []):
                    metric = str(job["metric"])
                    if metric in requested:
                        yield MetricTask(
                            audio_id=str(decision["audio_id"]),
                            path=Path(decision["path"]),
                            metric=metric,
                            device=str(job["device"]),
                        )
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid decision record at {path}:{line_no}: {exc}") from exc


def _create_runner(
    metric: str, device: str, utmos_checkpoint: Path | None, model_cache: Path
) -> MetricRunner:
    if metric == "mos":
        assert utmos_checkpoint is not None
        return UtmosV2Runner(utmos_checkpoint, device)
    if metric == "speaker_similarity":
        return EcapaRunner(model_cache / "ecapa", device)
    raise ValueError(f"No executable worker is registered for metric: {metric}")


def _torch_device(device: str) -> str:
    if device == "gpu":
        return "cuda:0"
    if device in {"cpu", "cuda", "cuda:0"}:
        return device
    raise ValueError(f"Unsupported device: {device}")


def _stable_seed(audio_id: str) -> int:
    return int.from_bytes(blake2s(audio_id.encode(), digest_size=4).digest(), "big")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
