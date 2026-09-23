from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import SignalFeatures


class Policy:
    def __init__(self, document: dict[str, Any]) -> None:
        self.document = document

    @classmethod
    def from_file(cls, path: Path) -> Policy:
        return cls(json.loads(path.read_text(encoding="utf-8")))

    @property
    def version(self) -> str:
        return str(self.document["version"])

    def evaluate(self, features: SignalFeatures) -> tuple[str, list[str]]:
        hard = self.document["hard_reject"]
        review = self.document["review"]
        reject_reasons = _violations(features, hard, prefix="hard")
        required_rates = self.document.get("input", {}).get("required_sample_rates_hz", [])
        if required_rates and features.sample_rate_hz not in required_rates:
            reject_reasons.append("hard:sample_rate_not_allowed")
        if self.document.get("input", {}).get("require_mono", False) and features.channels != 1:
            reject_reasons.append("hard:channel_count_not_allowed")
        if reject_reasons:
            return "reject", reject_reasons
        review_reasons = _violations(features, review, prefix="review")
        return ("review", review_reasons) if review_reasons else ("accept", [])


def _violations(features: SignalFeatures, thresholds: dict[str, float], prefix: str) -> list[str]:
    values = {
        "duration_s": features.duration_s,
        "rms_dbfs": features.rms_dbfs,
        "peak_dbfs": features.peak_dbfs,
        "dc_offset": abs(features.dc_offset),
        "clipping_ratio": features.clipping_ratio,
        "silence_ratio": features.silence_ratio,
    }
    reasons: list[str] = []
    for key, threshold in thresholds.items():
        if key.startswith("min_"):
            metric = key.removeprefix("min_")
            if values[metric] < threshold:
                reasons.append(f"{prefix}:{metric}_below_min")
        elif key.startswith("max_"):
            metric = key.removeprefix("max_")
            if values[metric] > threshold:
                reasons.append(f"{prefix}:{metric}_above_max")
    return reasons
