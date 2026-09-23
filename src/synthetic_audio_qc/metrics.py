from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetricSpec:
    devices: frozenset[str]
    default_device: str


METRICS: dict[str, MetricSpec] = {
    "vad": MetricSpec(frozenset({"cpu"}), "cpu"),
    "asr": MetricSpec(frozenset({"cpu", "gpu"}), "gpu"),
    "dnsmos": MetricSpec(frozenset({"cpu", "gpu"}), "cpu"),
    "mos": MetricSpec(frozenset({"cpu", "gpu"}), "gpu"),
    "speaker_similarity": MetricSpec(frozenset({"cpu", "gpu"}), "gpu"),
}
VALID_SCOPES = frozenset({"routed", "review", "non_rejected"})


def parse_metric_selection(value: str) -> tuple[str, ...]:
    selected = tuple(metric.strip() for metric in value.split(",") if metric.strip())
    if selected == ("all",):
        return tuple(METRICS)
    if "all" in selected:
        raise ValueError("Use 'all' alone, or provide an explicit comma-separated metric list")
    unknown = sorted(set(selected) - METRICS.keys())
    if unknown:
        raise ValueError(f"Unsupported metric(s): {', '.join(unknown)}")
    return tuple(dict.fromkeys(selected))


def validate_metric_policy(requested: tuple[str, ...], metric_policy: dict[str, Any]) -> None:
    for metric in requested:
        _configured_job(metric, metric_policy)


def scheduled_metric_jobs(
    status: str,
    route_heavy_model: bool,
    requested: tuple[str, ...],
    metric_policy: dict[str, Any],
) -> list[dict[str, str]]:
    """Create explicit worker jobs; this process never loads optional ML models."""
    if status == "reject":
        return []
    jobs: list[dict[str, str]] = []
    for metric in requested:
        device, scope = _configured_job(metric, metric_policy)
        eligible = {
            "routed": route_heavy_model,
            "review": status == "review",
            "non_rejected": True,
        }[scope]
        if eligible:
            jobs.append({"metric": metric, "device": device, "scope": scope})
    return jobs


def _configured_job(metric: str, metric_policy: dict[str, Any]) -> tuple[str, str]:
    spec = METRICS[metric]
    options = metric_policy.get(metric, {})
    device = options.get("device", spec.default_device)
    scope = options.get("scope", "routed")
    if device not in spec.devices:
        supported = ", ".join(sorted(spec.devices))
        raise ValueError(
            f"Metric '{metric}' does not support device '{device}' (supported: {supported})"
        )
    if scope not in VALID_SCOPES:
        allowed = ", ".join(sorted(VALID_SCOPES))
        raise ValueError(f"Metric '{metric}' has invalid scope '{scope}' (allowed: {allowed})")
    return device, scope
