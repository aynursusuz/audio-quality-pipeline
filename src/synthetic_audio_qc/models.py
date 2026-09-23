from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class AudioRecord:
    """One immutable input item. Metadata is deliberately opaque to the core checks."""

    audio_id: str
    path: str
    expected_text: str | None = None
    language: str | None = None
    generator: str | None = None
    generator_version: str | None = None
    voice_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalFeatures:
    sample_rate_hz: int
    channels: int
    frame_count: int
    duration_s: float
    peak_dbfs: float
    rms_dbfs: float
    dc_offset: float
    clipping_ratio: float
    silence_ratio: float
    sha256: str


@dataclass
class Decision:
    audio_id: str
    path: str
    status: str
    reasons: list[str]
    features: dict[str, Any] = field(default_factory=dict)
    duplicate_of: str | None = None
    route_heavy_model: bool = False
    route_human_audit: bool = False
    metric_jobs: list[dict[str, str]] = field(default_factory=list)
    pipeline_version: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.duplicate_of is None:
            payload.pop("duplicate_of")
        if not self.metric_jobs:
            payload.pop("metric_jobs")
        return payload
